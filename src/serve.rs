//! Shared axum/tokio server scaffold for MeiSei layer servers: env-driven
//! config, `/healthz`, and the platform-token-gated `/v1/mcp` entrypoint.
//! Each layer supplies a [`McpHandler`] that dispatches its own domain
//! methods; this module owns everything generic around it (routing, auth
//! wiring, the response envelope).

use std::future::Future;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;

use crate::auth::{self, Claims};

/// Static identity + build metadata for one layer server.
///
/// `default_version` and `git_sha` MUST be computed by the caller
/// (`env!("CARGO_PKG_VERSION")`, `option_env!("GIT_SHA").unwrap_or("dev")`)
/// in the layer's own `main.rs` — evaluating those macros inside layer-kit
/// would bake layer-kit's own version/build SHA into every layer's
/// `/healthz`, breaking the deploy drift-check that reads it.
pub struct ServeConfig {
    /// Tool name (`"torii"`, `"satori"`, ...). Also the token audience and
    /// the env var prefix: `{TOOL}_PORT` / `{TOOL}_PLATFORM_SECRET` /
    /// `{TOOL}_VERSION`.
    pub tool: &'static str,
    pub default_port: u16,
    pub default_version: &'static str,
    pub git_sha: &'static str,
}

/// A layer's MCP method dispatcher. `claims` is the verified platform-token
/// identity (workspace/project) for handlers that need to scope work (e.g.
/// satori's per-workspace semantic index); most handlers ignore it.
///
/// Spelled out as `-> impl Future<..> + Send` (rather than `async fn`) so the
/// returned future is provably `Send` — axum requires that to route the
/// method on a multi-threaded runtime. Implementors still just write
/// `async fn dispatch(..) { .. }`; that satisfies this signature as long as
/// the body's future is actually Send, which it is here.
pub trait McpHandler: Send + Sync + 'static {
    fn dispatch(
        &self,
        claims: &Claims,
        method: &str,
        params: serde_json::Value,
    ) -> impl Future<Output = Result<serde_json::Value, (StatusCode, serde_json::Value)>> + Send;
}

struct ServeState<H> {
    tool: &'static str,
    version: String,
    git_sha: &'static str,
    platform_secret: Option<Vec<u8>>,
    handler: H,
}

/// Run a layer server to completion: binds `127.0.0.1:<port>` (C3
/// hardening — only the co-located platform reaches it) and serves
/// `/healthz` + `/v1/mcp` until the process exits.
pub async fn serve<H: McpHandler>(config: ServeConfig, handler: H) {
    let ServeConfig {
        tool,
        default_port,
        default_version,
        git_sha,
    } = config;
    let prefix = tool.to_uppercase();

    let version = std::env::var(format!("{prefix}_VERSION"))
        .unwrap_or_else(|_| default_version.to_string());
    let platform_secret = std::env::var(format!("{prefix}_PLATFORM_SECRET"))
        .ok()
        .filter(|s| !s.is_empty())
        .map(String::into_bytes);
    if platform_secret.is_none() {
        tracing::warn!("{prefix}_PLATFORM_SECRET unset — /v1/mcp will reject all requests");
    }

    let state = Arc::new(ServeState {
        tool,
        version,
        git_sha,
        platform_secret,
        handler,
    });

    let app = Router::new()
        .route("/healthz", get(healthz::<H>))
        .route("/v1/mcp", post(mcp::<H>))
        .with_state(state);

    let port =
        std::env::var(format!("{prefix}_PORT")).unwrap_or_else(|_| default_port.to_string());
    // localhost-bound: only the co-located platform reaches it (C3 hardening).
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    tracing::info!(%addr, tool, "listening");
    axum::serve(listener, app).await.expect("server error");
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

async fn healthz<H: McpHandler>(State(s): State<Arc<ServeState<H>>>) -> impl IntoResponse {
    Json(json!({
        "service": s.tool,
        "status": "ok",
        "version": s.version,
        "git_sha": s.git_sha,
    }))
}

async fn mcp<H: McpHandler>(
    State(s): State<Arc<ServeState<H>>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let Some(secret) = &s.platform_secret else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error":"auth_disabled"})),
        )
            .into_response();
    };
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    let Some(claims) = token.and_then(|t| auth::verify(secret, s.tool, now_secs(), t)) else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error":"invalid_platform_token"})),
        )
            .into_response();
    };

    // Auth passed — dispatch the MCP method against the layer's own handler.
    let req: McpRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "bad_request", "detail": e.to_string()})),
            )
                .into_response();
        }
    };
    match s.handler.dispatch(&claims, &req.method, req.params).await {
        Ok(mut result) => {
            // Stamp the token-scoped call context onto the response envelope.
            result["tool"] = json!(s.tool);
            result["version"] = json!(s.version);
            result["workspace"] = json!(claims.workspace);
            result["project"] = json!(claims.project);
            Json(result).into_response()
        }
        Err((code, payload)) => (code, Json(payload)).into_response(),
    }
}

/// One MCP call: `{ "method": "<tool>.<op>", "params": { ... } }`.
#[derive(serde::Deserialize)]
struct McpRequest {
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}
