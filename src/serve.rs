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

    /// Tool descriptors for `tools/list` (`{name, description, inputSchema}`).
    /// Names are underscored (`torii_parse`) because MCP clients reject dots;
    /// [`dispatch_method`] maps them back onto dispatch methods
    /// (`torii.parse`).
    fn tools(&self) -> Vec<serde_json::Value> {
        Vec::new()
    }
}

/// `torii_parse` → `torii.parse`; anything else passes through unchanged
/// (a layer may already name its tools with dots).
fn dispatch_method(tool: &str, name: &str) -> String {
    match name.strip_prefix(&format!("{tool}_")) {
        Some(rest) => format!("{tool}.{rest}"),
        None => name.to_string(),
    }
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
    let req: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "bad_request", "detail": e.to_string()})),
            )
                .into_response();
        }
    };

    // MCP (JSON-RPC 2.0) is the only wire format — what Claude/Cursor and the
    // platform speak. The legacy `{method, params}` envelope is gone.
    if req.get("jsonrpc").is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "bad_request", "detail": "expected JSON-RPC 2.0 request"})),
        )
            .into_response();
    }
    match rpc(&s.handler, s.tool, &s.version, &claims, req).await {
        Some(resp) => Json(resp).into_response(),
        // Notification (no id): nothing to answer.
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// MCP protocol revision this scaffold implements.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// Handle one JSON-RPC frame. `None` = notification (nothing to send back).
/// Protocol-level failures come back as JSON-RPC errors with HTTP 200, per
/// the spec; only auth failures are HTTP errors.
async fn rpc<H: McpHandler>(
    handler: &H,
    tool: &'static str,
    version: &str,
    claims: &Claims,
    req: serde_json::Value,
) -> Option<serde_json::Value> {
    let id = req.get("id").filter(|v| !v.is_null()).cloned()?;
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
    let reply = |result| json!({"jsonrpc": "2.0", "id": id, "result": result});
    let fail = |code: i32, message: String| {
        json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
    };

    Some(match method {
        "initialize" => reply(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": tool, "version": version},
        })),
        "ping" => reply(json!({})),
        "tools/list" => reply(json!({"tools": handler.tools()})),
        "tools/call" => {
            let Some(name) = params.get("name").and_then(|n| n.as_str()) else {
                return Some(fail(-32602, "tools/call requires `name`".into()));
            };
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            // A tool that fails is a result with isError, not a protocol error:
            // the model must see the failure to react to it.
            let (payload, is_error) =
                match handler.dispatch(claims, &dispatch_method(tool, name), args).await {
                    Ok(result) => (result, false),
                    Err((_, err)) => (err, true),
                };
            reply(json!({
                "content": [{"type": "text", "text": payload.to_string()}],
                "structuredContent": payload,
                "isError": is_error,
            }))
        }
        other => fail(-32601, format!("method not found: {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;

    impl McpHandler for Fake {
        async fn dispatch(
            &self,
            _claims: &Claims,
            method: &str,
            params: serde_json::Value,
        ) -> Result<serde_json::Value, (StatusCode, serde_json::Value)> {
            match method {
                "test.echo" => Ok(json!({"echo": params})),
                other => Err((
                    StatusCode::BAD_REQUEST,
                    json!({"error": "unknown_method", "detail": other}),
                )),
            }
        }

        fn tools(&self) -> Vec<serde_json::Value> {
            vec![json!({"name": "test_echo", "description": "echo", "inputSchema": {"type": "object"}})]
        }
    }

    fn claims() -> Claims {
        Claims {
            workspace: "ws1".into(),
            project: None,
            tool: "test".into(),
            exp: 0,
        }
    }

    async fn call(req: serde_json::Value) -> Option<serde_json::Value> {
        rpc(&Fake, "test", "0.1.0", &claims(), req).await
    }

    #[tokio::test]
    async fn handshake_list_and_call() {
        let init = call(json!({"jsonrpc":"2.0","id":1,"method":"initialize"}))
            .await
            .unwrap();
        assert_eq!(init["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(init["result"]["serverInfo"]["name"], "test");

        let list = call(json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}))
            .await
            .unwrap();
        assert_eq!(list["result"]["tools"][0]["name"], "test_echo");

        // Underscored tool name must reach the dotted dispatch method.
        let out = call(json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                              "params":{"name":"test_echo","arguments":{"x":1}}}))
            .await
            .unwrap();
        assert_eq!(out["result"]["isError"], false);
        assert_eq!(out["result"]["structuredContent"]["echo"]["x"], 1);
        assert!(out["result"]["content"][0]["text"].as_str().unwrap().contains("\"x\":1"));
    }

    #[tokio::test]
    async fn notification_unknown_method_and_tool_failure() {
        assert!(
            call(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
                .await
                .is_none(),
            "notifications get no response"
        );

        let bad = call(json!({"jsonrpc":"2.0","id":1,"method":"nope"}))
            .await
            .unwrap();
        assert_eq!(bad["error"]["code"], -32601);

        // A failing tool is a result with isError, not a JSON-RPC error.
        let failed = call(json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
                                 "params":{"name":"test_missing"}}))
            .await
            .unwrap();
        assert_eq!(failed["result"]["isError"], true);
        assert_eq!(failed["result"]["structuredContent"]["error"], "unknown_method");
    }
}
