//! Concrete AI provider for a layer server: a minimal client for the OpenAI
//! Responses, OpenAI Chat Completions and Anthropic Messages APIs,
//! implementing the [`crate::ai::AiProvider`] seam.
//!
//! Config (env), mirroring the daruma ai-infra contract so one key serves
//! the whole co-located platform:
//!   OPENAI_API_KEY  — required; when unset/empty `AiConfig::from_env`
//!                     returns `None` and AI methods answer `ai_not_configured`
//!                     instead of failing at call time.
//!   OPENAI_BASE_URL — default `https://api.openai.com/v1`
//!                     (`https://api.anthropic.com` for `anthropic_messages`).
//!   OPENAI_MODEL    — default `gpt-4.1`.
//!   OPENAI_API_PROTOCOL — `responses` (default), `chat_completions` or
//!                     `anthropic_messages`; an unknown value disables AI.
//!
//! Transport timeouts and keepalive mirror daruma's ai-infra client so one
//! wedged upstream provider cannot stall a layer hop forever:
//!   OPENAI_REQUEST_TIMEOUT_SECONDS — optional cap on a whole request,
//!                     default [`REQUEST_TIMEOUT`].

use serde_json::{json, Value};
use std::time::Duration;

pub use crate::ai::AiConfig;
use crate::ai::ApiProtocol;
use crate::ai::{parse_usage, AiError, AiOutput, AiProvider, AiRequest, AiUsage, ToolCall};

/// [`AiProvider`] backed by the OpenAI Responses API. Clone is cheap (the
/// inner `reqwest::Client` is Arc-backed).
#[derive(Clone)]
pub struct OpenAiProvider {
    http: reqwest::Client,
    cfg: AiConfig,
}

/// How long to wait for the TCP+TLS handshake before giving up.
///
/// The kernel's own ceiling here is `tcp_syn_retries`, typically two minutes of
/// silent retrying. Nothing upstream wants to wait that long to learn a provider
/// is unreachable.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Ceiling on a whole request, handshake to last byte.
///
/// Well above any legitimate reasoning time for a single tool call, and far
/// enough below "wait forever" to bound the damage when a provider accepts a
/// request and never answers. That is not hypothetical: production recorded a
/// call sitting on a TCP-healthy connection for a full 300 seconds with nothing
/// coming back, which no transport setting can fix. This is the ceiling that
/// actually applies now that `tcp_user_timeout` no longer cuts calls off early,
/// so it doubles as the cap on how long one wedged call can stall an audit pass.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Keepalive probe interval on idle sockets.
///
/// A request awaiting a slow model looks exactly like an idle connection to a
/// NAT or load balancer, which is how such a connection gets dropped mid-answer;
/// probes keep the flow alive and, if the peer really is gone, surface it as a
/// prompt error rather than a stalled read.
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

/// Drop pooled connections well before typical middlebox idle limits, so a
/// request is not handed a socket that has already been discarded upstream.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

impl OpenAiProvider {
    pub fn new(cfg: AiConfig) -> Self {
        let timeout = cfg
            .request_timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(REQUEST_TIMEOUT);
        let http = Self::http_client_builder()
            .timeout(timeout)
            .tcp_user_timeout(timeout)
            .build()
            .expect("openai http client build failed");
        Self::with_http_client(cfg, http)
    }

    /// Build a provider over a caller-supplied HTTP client.
    ///
    /// [`OpenAiProvider::new`] applies the documented transport budget itself;
    /// this constructor is for callers (tests, hosts with their own TLS or
    /// proxy policy) that must control the client directly. Whatever timeouts
    /// the client carries are the ones that apply — the provider adds none.
    pub fn with_http_client(cfg: AiConfig, http: reqwest::Client) -> Self {
        Self { http, cfg }
    }

    /// The transport settings [`OpenAiProvider::new`] applies.
    ///
    /// Exposed so callers that must build their own client start from these
    /// rather than from a bare `reqwest::Client::new()`, which has no timeout,
    /// no connect timeout and no keepalive whatsoever.
    pub fn http_client_builder() -> reqwest::ClientBuilder {
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .tcp_keepalive(TCP_KEEPALIVE)
            // `reqwest` defaults this to 30 seconds, which quietly caps how long
            // any request may wait: `TCP_USER_TIMEOUT` makes the kernel abort the
            // connection with `ETIMEDOUT` once data goes unacknowledged that
            // long, and an unanswered keepalive probe counts. A provider that
            // stays silent while its model thinks therefore had its connection
            // killed mid-answer at ~32s — measured in production at 32348,
            // 32102 and 32263 ms, a spread far too tight for packet loss.
            //
            // The ceiling on a slow model belongs to the request timeout, not to
            // a TCP-level abort a full order of magnitude below it, so this is
            // aligned with `REQUEST_TIMEOUT`. Keepalive still runs, so a peer
            // that is genuinely gone is still detected — just not mistaken for
            // one that is merely thinking.
            .tcp_user_timeout(REQUEST_TIMEOUT)
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        // Deliberately no `CryptoProvider::install_default()` here. layer-kit
        // builds reqwest with the `rustls-tls` feature, whose ring provider is
        // compiled in and selected automatically, so a manual install is
        // redundant. daruma's ai-infra does install one because it builds with
        // `rustls-no-provider`; copying that setup here would drag in
        // tokio-rustls as an extra dependency for nothing. If this crate ever
        // switches to a no-provider feature set, revisit.
    }

    pub fn model(&self) -> &str {
        &self.cfg.model
    }
}

impl AiProvider for OpenAiProvider {
    async fn respond(&self, req: AiRequest) -> Result<Vec<AiOutput>, AiError> {
        Ok(self.respond_with_usage(req).await?.0)
    }

    async fn respond_with_usage(
        &self,
        req: AiRequest,
    ) -> Result<(Vec<AiOutput>, Option<AiUsage>), AiError> {
        let (url, body) = endpoint_and_body(&self.cfg, &req)?;
        let mut request = self.http.post(url);
        for (name, value) in auth_headers(&self.cfg) {
            request = request.header(name, value);
        }
        let resp = request
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::new(format!("ai request failed: {e}")))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let message = resp.text().await.unwrap_or_default();
            return Err(AiError::new(format!("ai api status {status}: {message}")));
        }
        let body: Value = resp
            .json()
            .await
            // A body that fails to read or decode is a transport fault (cut
            // connection, proxy HTML page), not a schema the model violated:
            // classing it Schema sent fujin into a repair retry on a network
            // error.
            .map_err(|e| AiError::new(format!("ai response decode failed: {e}")))?;
        let outputs = match self.cfg.api_protocol {
            ApiProtocol::Responses => parse_outputs(&body)?,
            ApiProtocol::ChatCompletions => parse_chat_outputs(&body)?,
            ApiProtocol::AnthropicMessages => parse_anthropic_outputs(&body)?,
        };
        let mut usage = parse_usage(&body);
        // Anthropic reports input/output only; the sum is what the others call total.
        if let Some(usage) = usage.as_mut() {
            if usage.total_tokens.is_none() {
                if let (Some(input), Some(output)) = (usage.input_tokens, usage.output_tokens) {
                    usage.total_tokens = Some(input + output);
                }
            }
        }
        Ok((outputs, usage))
    }
}

/// Anthropic API version pinned in the `anthropic-version` header.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Authentication headers for the configured protocol. OpenAI-style
/// providers take `Authorization: Bearer`; Anthropic rejects it and wants
/// `x-api-key` plus a pinned `anthropic-version` instead.
fn auth_headers(cfg: &AiConfig) -> Vec<(&'static str, String)> {
    match cfg.api_protocol {
        ApiProtocol::Responses | ApiProtocol::ChatCompletions => {
            vec![("authorization", format!("Bearer {}", cfg.api_key))]
        }
        ApiProtocol::AnthropicMessages => vec![
            ("x-api-key", cfg.api_key.clone()),
            ("anthropic-version", ANTHROPIC_VERSION.to_owned()),
        ],
    }
}

/// Endpoint + matching body for the configured protocol. Split out so a unit
/// test can prove the pairing without a network: crossed arms here (right URL,
/// wrong body) are exactly the outage where every layer posted Responses
/// bodies to a Chat-Completions-only provider.
fn endpoint_and_body(cfg: &AiConfig, req: &AiRequest) -> Result<(String, Value), AiError> {
    Ok(match cfg.api_protocol {
        ApiProtocol::Responses => (cfg.responses_url(), build_request_body(cfg, req)),
        ApiProtocol::ChatCompletions => (
            cfg.chat_completions_url(),
            build_chat_request_body(cfg, req),
        ),
        ApiProtocol::AnthropicMessages => (
            cfg.anthropic_messages_url(),
            build_anthropic_request_body(cfg, req)?,
        ),
    })
}

/// Default `max_output_tokens` when the config does not set one. Layer
/// operations return short structured tool-calls; a couple thousand tokens is
/// generous while keeping proxy billers' cost forecast (and thus the balance
/// reserve) small.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 2000;

/// OpenAI strict tools require every object to reject undeclared keys and to
/// list every declared property as required. Callers opt in with
/// `"strict": true`; tools without that flag are left untouched for
/// OpenAI-compatible providers that do not implement strict schemas.
fn prepare_tool(tool: &Value) -> Value {
    let mut tool = tool.clone();
    if tool.get("strict").and_then(Value::as_bool) == Some(true) {
        if let Some(schema) = tool.get_mut("parameters") {
            make_schema_strict(schema);
        }
    }
    tool
}

fn make_schema_strict(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if obj.get("type").and_then(Value::as_str) == Some("object") {
        let required = obj
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.keys().cloned().map(Value::String).collect());
        obj.insert("additionalProperties".into(), Value::Bool(false));
        if let Some(required) = required {
            obj.insert("required".into(), Value::Array(required));
        }
    }
    for value in obj.values_mut() {
        match value {
            Value::Object(_) => make_schema_strict(value),
            Value::Array(values) => values.iter_mut().for_each(make_schema_strict),
            _ => {}
        }
    }
}

/// Build the Responses API request body. Pure — unit-tested without network.
fn build_request_body(cfg: &AiConfig, req: &AiRequest) -> Value {
    let mut obj = json!({
        "model": cfg.model,
        "input": req.input,
        "max_output_tokens": cfg.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
    });
    if !req.tools.is_empty() {
        obj["tools"] = Value::Array(req.tools.iter().map(prepare_tool).collect());
    }
    if let Some(tc) = &req.tool_choice {
        obj["tool_choice"] = Value::String(tc.clone());
    }
    obj
}

/// Build the Chat Completions request body from the same provider-neutral
/// request: `input` becomes the sole user message, Responses-style flat tool
/// schemas are wrapped into `{"type":"function","function":{...}}`.
fn build_chat_request_body(cfg: &AiConfig, req: &AiRequest) -> Value {
    let mut obj = json!({
        "model": cfg.model,
        "messages": [{"role": "user", "content": req.input}],
        "max_tokens": cfg.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
    });
    if !req.tools.is_empty() {
        obj["tools"] = Value::Array(
            req.tools
                .iter()
                .map(|tool| {
                    let tool = prepare_tool(tool);
                    let mut function = serde_json::Map::new();
                    for field in ["name", "description", "parameters", "strict"] {
                        if let Some(value) = tool.get(field) {
                            function.insert(field.into(), value.clone());
                        }
                    }
                    json!({"type": "function", "function": function})
                })
                .collect(),
        );
    }
    if let Some(tc) = &req.tool_choice {
        obj["tool_choice"] = Value::String(tc.clone());
    }
    obj
}

/// Build the Anthropic Messages request body from the same provider-neutral
/// request. Pure.
///
/// - A string `input` becomes the single user message. An array of
///   `{role, content}` messages is accepted too: `system` entries are lifted
///   into the top-level `system` field (Messages API has no system role), the
///   rest stay messages. Any other shape is refused rather than sent as an
///   invalid `content`.
/// - `max_tokens` is mandatory in this API, so it is always sent.
/// - Function tools become `{name, description, input_schema}`; a `strict`
///   opt-in still closes the schema, but the flag itself is not sent.
/// - `tool_choice` maps `auto`/`required`/`none` and a tool name; it is
///   omitted without tools — there is nothing to choose.
///
/// Prompt caching and batches are deliberately not ported from ai-infra.
fn build_anthropic_request_body(cfg: &AiConfig, req: &AiRequest) -> Result<Value, AiError> {
    let (system, messages) = anthropic_messages(&req.input)?;
    let mut obj = json!({
        "model": cfg.model,
        "max_tokens": cfg.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
        "messages": messages,
    });
    if !system.is_empty() {
        obj["system"] = Value::String(system.join("\n\n"));
    }
    if req.tools.is_empty() {
        return Ok(obj);
    }
    obj["tools"] = Value::Array(
        req.tools
            .iter()
            .map(|tool| {
                let tool = prepare_tool(tool);
                let name = tool.get("name").and_then(Value::as_str).ok_or_else(|| {
                    AiError::new("anthropic_messages function tool requires name")
                })?;
                let mut out = json!({
                    "name": name,
                    // Required by Anthropic; a parameterless function has none.
                    "input_schema": tool
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                });
                if let Some(description) = tool.get("description") {
                    out["description"] = description.clone();
                }
                Ok(out)
            })
            .collect::<Result<Vec<_>, AiError>>()?,
    );
    if let Some(tc) = &req.tool_choice {
        obj["tool_choice"] = match tc.as_str() {
            "auto" => json!({"type": "auto"}),
            "required" => json!({"type": "any"}),
            "none" => json!({"type": "none"}),
            name => json!({"type": "tool", "name": name}),
        };
    }
    Ok(obj)
}

/// Split a provider-neutral `input` into Anthropic's `system` texts and
/// `messages`. See [`build_anthropic_request_body`] for the accepted shapes.
fn anthropic_messages(input: &Value) -> Result<(Vec<String>, Vec<Value>), AiError> {
    if let Some(text) = input.as_str() {
        return Ok((vec![], vec![json!({"role": "user", "content": text})]));
    }
    let unsupported = || {
        AiError::new(
            "anthropic_messages supports a string input or an array of {role, content} messages",
        )
    };
    let items = input.as_array().ok_or_else(unsupported)?;
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for item in items {
        let role = item["role"].as_str().ok_or_else(unsupported)?;
        let content = item.get("content").ok_or_else(unsupported)?;
        if role == "system" || role == "developer" {
            system.push(content.as_str().ok_or_else(unsupported)?.to_owned());
        } else {
            messages.push(json!({"role": role, "content": content}));
        }
    }
    if messages.is_empty() {
        return Err(AiError::new("anthropic_messages input has no user message"));
    }
    Ok((system, messages))
}

/// Parse an Anthropic Messages reply into lib [`AiOutput`]s. Pure.
fn parse_anthropic_outputs(body: &Value) -> Result<Vec<AiOutput>, AiError> {
    // A `max_tokens` stop leaves a half-written `tool_use.input`: name the
    // setting instead of passing a truncated call on.
    if body["stop_reason"] == "max_tokens" {
        return Err(AiError::output_budget(
            "response incomplete (max_tokens); increase max_output_tokens or ask for fewer items",
        ));
    }
    let blocks = body["content"]
        .as_array()
        .ok_or_else(|| AiError::schema("response missing 'content' array"))?;
    let mut out = Vec::new();
    for block in blocks {
        match block["type"].as_str() {
            Some("text") => {
                if let Some(text) = block["text"].as_str() {
                    out.push(AiOutput::Text(text.to_owned()));
                }
            }
            Some("tool_use") => {
                let name = block["name"]
                    .as_str()
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| AiError::schema("tool_use block missing name"))?;
                // Anthropic sends `input` as an object; ToolCall carries the
                // OpenAI-style JSON string that callers `from_str`.
                let input = block
                    .get("input")
                    .filter(|input| input.is_object())
                    .ok_or_else(|| AiError::schema("tool_use block input must be a JSON object"))?;
                out.push(AiOutput::ToolCall(ToolCall {
                    name: name.to_owned(),
                    arguments: input.to_string(),
                }));
            }
            _ => {} // thinking and future block types carry nothing callers use.
        }
    }
    if out.is_empty() {
        return Err(AiError::schema(
            "anthropic response has neither text nor tool_use content",
        ));
    }
    Ok(out)
}

/// Parse a Chat Completions reply into lib [`AiOutput`]s. Pure.
fn parse_chat_outputs(body: &Value) -> Result<Vec<AiOutput>, AiError> {
    let choice = body["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .ok_or_else(|| AiError::schema("response missing 'choices[0]'"))?;
    if choice["finish_reason"] == "length" {
        return Err(AiError::output_budget(
            "chat completion exhausted its token budget; increase max_output_tokens",
        ));
    }
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| AiError::schema("response missing 'choices[0].message'"))?;
    let mut out = Vec::new();
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        out.push(AiOutput::Text(content.to_owned()));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let function = &tool_call["function"];
            let name = function["name"]
                .as_str()
                .ok_or_else(|| AiError::schema("tool call missing function.name"))?;
            let arguments = function["arguments"].as_str().ok_or_else(|| {
                AiError::schema("tool call function.arguments must be a JSON string")
            })?;
            serde_json::from_str::<Value>(arguments).map_err(|error| {
                AiError::schema(format!(
                    "invalid tool call function.arguments JSON: {error}"
                ))
            })?;
            out.push(AiOutput::ToolCall(ToolCall {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            }));
        }
    }
    if out.is_empty() {
        return Err(AiError::schema(
            "chat response has neither content nor tool_calls",
        ));
    }
    Ok(out)
}

/// Parse the `output` array of a Responses API reply into lib [`AiOutput`]s.
/// Pure — unit-tested without network.
fn parse_outputs(body: &Value) -> Result<Vec<AiOutput>, AiError> {
    if body["status"] == "incomplete" {
        let reason = body["incomplete_details"]["reason"]
            .as_str()
            .unwrap_or("unknown");
        let message = format!(
            "response incomplete ({reason}); increase max_output_tokens or ask for fewer items"
        );
        return Err(if reason == "max_output_tokens" {
            AiError::output_budget(message)
        } else {
            AiError::schema(message)
        });
    }
    let items = body["output"]
        .as_array()
        .ok_or_else(|| AiError::schema("response missing 'output' array"))?;
    let mut out = Vec::new();
    for item in items {
        match item["type"].as_str() {
            Some("message") => {
                if let Some(content) = item["content"].as_array() {
                    for part in content {
                        if part["type"] == "output_text" {
                            if let Some(text) = part["text"].as_str() {
                                out.push(AiOutput::Text(text.to_owned()));
                            }
                        }
                    }
                }
            }
            Some("function_call") => {
                let name = item["name"]
                    .as_str()
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| AiError::schema("function call missing name"))?;
                let arguments = item["arguments"].as_str().ok_or_else(|| {
                    AiError::schema("function call arguments must be a JSON string")
                })?;
                serde_json::from_str::<Value>(arguments).map_err(|error| {
                    AiError::schema(format!("invalid function call arguments JSON: {error}"))
                })?;
                out.push(AiOutput::ToolCall(ToolCall {
                    name: name.to_owned(),
                    arguments: arguments.to_owned(),
                }));
            }
            _ => {} // Unknown output type — skip gracefully.
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::AiErrorKind;

    fn cfg(max_output_tokens: Option<u32>) -> AiConfig {
        AiConfig {
            api_key: "sk-test".into(),
            base_url: "https://api.example.com/v1".into(),
            model: "gpt-4.1".into(),
            max_output_tokens,
            api_protocol: ApiProtocol::default(),
            request_timeout_seconds: None,
        }
    }

    #[test]
    fn transport_constants_match_the_documented_budget() {
        // The values are load-bearing (see the doc comments above and the
        // production incident they encode); lock them so an innocent-looking
        // bump cannot silently change the failure mode of a wedged provider.
        assert_eq!(CONNECT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(REQUEST_TIMEOUT, Duration::from_secs(120));
        assert_eq!(TCP_KEEPALIVE, Duration::from_secs(30));
        assert_eq!(POOL_IDLE_TIMEOUT, Duration::from_secs(60));
    }

    #[test]
    fn http_client_builder_builds_a_client() {
        // The shared builder must stay buildable on its own: callers that
        // need their own client start from it instead of Client::new().
        let client = OpenAiProvider::http_client_builder().build();
        assert!(client.is_ok(), "{client:?}");
    }

    /// Регрессия на «зависший upstream вешает хоп навечно»: провайдер,
    /// построенный против TCP-сокета, который принимает соединение и молчит,
    /// обязан вернуть ошибку в пределах таймаута запроса, а не висеть.
    #[tokio::test]
    async fn black_hole_provider_fails_within_request_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // Чёрная дыра: accept'им и держим сокет открытым, но не отвечаем.
        // ВАЖНО: сокет обязан быть именно привязан к переменной и доживать
        // вместе с задачей — `let _ = socket` роняет его сразу (паттерн `_`
        // не биндит), клиент получает мгновенный FIN вместо тишины.
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let _held_open = socket; // держим, ответа не будет
                    std::future::pending::<()>().await;
                });
            }
        });

        let mut config = cfg(None);
        config.base_url = format!("http://{addr}/v1");
        // Тот же механизм, что и OPENAI_REQUEST_TIMEOUT_SECONDS, но с
        // маленьким значением — тест не должен ждать дефолтные 120 секунд.
        config.request_timeout_seconds = Some(1);
        // Клиент собираем вручную с no_proxy(): иначе на машинах с
        // HTTP_PROXY/HTTPS_PROXY reqwest уходит через системный прокси,
        // который мгновенно отвечает 502, и вместо таймаута тест получает
        // быструю ошибку — ассерт «провисел >= 500 мс» падает. no_proxy()
        // отключает системные прокси, делая чёрную дыру единственным путём.
        let http = OpenAiProvider::http_client_builder()
            .no_proxy()
            .timeout(Duration::from_secs(1))
            .tcp_user_timeout(Duration::from_secs(1))
            .build()
            .expect("test http client build failed");
        let provider = OpenAiProvider::with_http_client(config, http);

        let started = std::time::Instant::now();
        let err = provider
            .respond_with_usage(AiRequest {
                input: Value::String("ping".into()),
                tools: vec![],
                tool_choice: None,
            })
            .await
            .expect_err("black-hole upstream must fail within the timeout, not hang");
        let elapsed = started.elapsed();

        assert!(
            elapsed <= Duration::from_secs(5),
            "call stalled {elapsed:?} past the 1s request timeout: {err}"
        );
        // Соединение установлено успешно, значит ошибка пришла именно от
        // таймаута ожидания ответа, а не от refused-коннекта.
        assert!(
            elapsed >= Duration::from_millis(500),
            "failed too early ({elapsed:?}) — not the request timeout: {err}"
        );
    }

    /// L-11: a 200 whose body is not JSON (a proxy's HTML page, a cut
    /// connection) is a transport fault. It must be `Other`, not `Schema`,
    /// or fujin's schema repair retries a network error.
    #[tokio::test]
    async fn undecodable_body_is_other_not_schema() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    let _ = socket.read(&mut buf).await;
                    let body = "<html>bad gateway</html>";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                });
            }
        });

        let mut config = cfg(None);
        config.base_url = format!("http://{addr}/v1");
        // no_proxy(): the dev host's HTTP_PROXY would otherwise answer for
        // the loopback stub (see black_hole_provider_fails_within_request_timeout).
        let http = OpenAiProvider::http_client_builder()
            .no_proxy()
            .build()
            .expect("test http client build failed");
        let provider = OpenAiProvider::with_http_client(config, http);

        let err = provider
            .respond_with_usage(AiRequest {
                input: Value::String("ping".into()),
                tools: vec![],
                tool_choice: None,
            })
            .await
            .expect_err("non-JSON body must fail");
        assert_eq!(err.kind(), AiErrorKind::Other, "{err}");
        assert!(
            err.to_string().starts_with("ai response decode failed"),
            "{err}"
        );
    }

    #[test]
    fn build_body_minimal_and_with_tools() {
        let req = AiRequest {
            input: Value::String("hello".into()),
            tools: vec![],
            tool_choice: None,
        };
        let body = build_request_body(&cfg(None), &req);
        assert_eq!(body["model"], "gpt-4.1");
        assert_eq!(body["input"], "hello");
        assert_eq!(body["max_output_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());

        let req = AiRequest {
            input: Value::String("p".into()),
            tools: vec![json!({"type": "function", "name": "some_tool"})],
            tool_choice: Some("required".into()),
        };
        let body = build_request_body(&cfg(Some(512)), &req);
        assert_eq!(body["max_output_tokens"], 512);
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"][0]["name"], "some_tool");
    }

    #[test]
    fn strict_tools_keep_the_flag_and_close_nested_schemas_in_both_protocols() {
        let req = AiRequest {
            input: Value::String("p".into()),
            tools: vec![json!({
                "type": "function",
                "name": "report",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}}
                        }
                    }
                }
            })],
            tool_choice: Some("required".into()),
        };

        let responses = build_request_body(&cfg(None), &req);
        let responses_tool = &responses["tools"][0];
        assert_eq!(responses_tool["strict"], true);
        assert_eq!(responses_tool["parameters"]["additionalProperties"], false);
        assert_eq!(responses_tool["parameters"]["required"], json!(["result"]));
        assert_eq!(
            responses_tool["parameters"]["properties"]["result"]["additionalProperties"],
            false
        );
        assert_eq!(
            responses_tool["parameters"]["properties"]["result"]["required"],
            json!(["ok"])
        );

        let mut chat_cfg = cfg(None);
        chat_cfg.api_protocol = ApiProtocol::ChatCompletions;
        let chat = build_chat_request_body(&chat_cfg, &req);
        let chat_function = &chat["tools"][0]["function"];
        assert_eq!(chat_function["strict"], responses_tool["strict"]);
        assert_eq!(chat_function["parameters"], responses_tool["parameters"]);
    }

    #[test]
    fn tools_without_strict_opt_in_keep_provider_compatible_schema() {
        let tool = json!({
            "type": "function",
            "name": "report",
            "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        });
        let req = AiRequest {
            input: Value::String("p".into()),
            tools: vec![tool.clone()],
            tool_choice: None,
        };

        assert_eq!(build_request_body(&cfg(None), &req)["tools"][0], tool);
        let chat = build_chat_request_body(&cfg(None), &req);
        assert!(chat["tools"][0]["function"].get("strict").is_none());
        assert!(chat["tools"][0]["function"]["parameters"]
            .get("additionalProperties")
            .is_none());
    }

    #[test]
    fn parse_outputs_message_and_function_call() {
        let body = json!({"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "name": "some_tool", "arguments": "{\"a\":1}"}
        ]});
        let out = parse_outputs(&body).unwrap();
        assert!(matches!(&out[0], AiOutput::Text(t) if t == "hi"));
        assert!(matches!(&out[1], AiOutput::ToolCall(tc) if tc.name == "some_tool"));
    }

    #[test]
    fn parse_outputs_missing_array_is_error() {
        assert_eq!(
            parse_outputs(&json!({"id": "resp_1"})).unwrap_err().kind(),
            AiErrorKind::Schema
        );
    }

    #[test]
    fn parse_outputs_classifies_budget_exhaustion_and_malformed_arguments() {
        let incomplete = json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "function_call", "name": "report", "arguments": "{\"ok\":"}]
        });
        assert_eq!(
            parse_outputs(&incomplete).unwrap_err().kind(),
            AiErrorKind::OutputBudget
        );

        let malformed = json!({
            "status": "completed",
            "output": [{"type": "function_call", "name": "report", "arguments": "{\"ok\":"}]
        });
        assert_eq!(
            parse_outputs(&malformed).unwrap_err().kind(),
            AiErrorKind::Schema
        );
    }

    #[test]
    fn protocol_picks_matching_endpoint_and_body() {
        let req = AiRequest {
            input: Value::String("p".into()),
            tools: vec![json!({"type": "function", "name": "some_tool",
                              "parameters": {"type": "object"}})],
            tool_choice: Some("required".into()),
        };

        let (url, body) = endpoint_and_body(&cfg(None), &req).unwrap();
        assert!(url.ends_with("/responses"), "{url}");
        assert_eq!(body["input"], "p");
        assert_eq!(body["tools"][0]["name"], "some_tool");

        let mut chat_cfg = cfg(Some(512));
        chat_cfg.api_protocol = ApiProtocol::ChatCompletions;
        let (url, body) = endpoint_and_body(&chat_cfg, &req).unwrap();
        assert!(url.ends_with("/chat/completions"), "{url}");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "p");
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "some_tool");
        assert!(body.get("input").is_none());
        assert!(body.get("max_output_tokens").is_none());

        let mut anthropic_cfg = cfg(Some(256));
        anthropic_cfg.api_protocol = ApiProtocol::AnthropicMessages;
        anthropic_cfg.base_url = "https://api.anthropic.com".into();
        let (url, body) = endpoint_and_body(&anthropic_cfg, &req).unwrap();
        assert_eq!(url, "https://api.anthropic.com/v1/messages");
        assert_eq!(body["messages"], json!([{"role": "user", "content": "p"}]));
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["tools"][0]["name"], "some_tool");
        assert_eq!(body["tools"][0]["input_schema"], json!({"type": "object"}));
        assert_eq!(body["tool_choice"], json!({"type": "any"}));
        assert!(body.get("input").is_none());
        assert!(body.get("max_output_tokens").is_none());
        assert!(body.get("system").is_none());
    }

    #[test]
    fn anthropic_url_does_not_double_the_version() {
        let mut config = cfg(None);
        config.api_protocol = ApiProtocol::AnthropicMessages;
        for base in [
            "https://api.anthropic.com",
            "https://api.anthropic.com/",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/",
        ] {
            config.base_url = base.into();
            assert_eq!(
                config.anthropic_messages_url(),
                "https://api.anthropic.com/v1/messages",
                "{base}"
            );
        }
    }

    #[test]
    fn auth_headers_follow_the_protocol() {
        let mut config = cfg(None);
        assert_eq!(
            auth_headers(&config),
            vec![("authorization", "Bearer sk-test".to_owned())]
        );
        config.api_protocol = ApiProtocol::AnthropicMessages;
        let headers = auth_headers(&config);
        assert_eq!(
            headers,
            vec![
                ("x-api-key", "sk-test".to_owned()),
                ("anthropic-version", "2023-06-01".to_owned()),
            ]
        );
        assert!(headers.iter().all(|(name, _)| *name != "authorization"));
    }

    #[test]
    fn anthropic_body_lifts_system_and_maps_tools() {
        let mut config = cfg(None);
        config.api_protocol = ApiProtocol::AnthropicMessages;
        let req = AiRequest {
            input: json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]),
            tools: vec![json!({
                "type": "function",
                "name": "report",
                "description": "d",
                "strict": true,
                "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}}
            })],
            tool_choice: Some("report".into()),
        };
        let body = build_anthropic_request_body(&config, &req).unwrap();
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
        assert_eq!(body["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS);
        let tool = &body["tools"][0];
        assert_eq!(tool["description"], "d");
        assert_eq!(tool["input_schema"]["additionalProperties"], false);
        assert_eq!(tool["input_schema"]["required"], json!(["ok"]));
        assert!(tool.get("strict").is_none());
        assert!(tool.get("parameters").is_none());
        assert_eq!(
            body["tool_choice"],
            json!({"type": "tool", "name": "report"})
        );

        // Without tools there is nothing to choose: tool_choice is omitted.
        let bare = AiRequest {
            input: Value::String("hi".into()),
            tools: vec![],
            tool_choice: Some("required".into()),
        };
        let body = build_anthropic_request_body(&config, &bare).unwrap();
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("tools").is_none());

        // A shape the Messages API cannot carry is refused, not sent.
        let odd = AiRequest {
            input: json!({"prompt": "hi"}),
            tools: vec![],
            tool_choice: None,
        };
        assert!(build_anthropic_request_body(&config, &odd).is_err());
        let system_only = AiRequest {
            input: json!([{"role": "system", "content": "s"}]),
            tools: vec![],
            tool_choice: None,
        };
        assert!(build_anthropic_request_body(&config, &system_only).is_err());
    }

    #[test]
    fn parse_anthropic_text_and_tool_use() {
        let body = json!({
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "…"},
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "toolu_1", "name": "report", "input": {"ok": true}}
            ]
        });
        let out = parse_anthropic_outputs(&body).unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(&out[0], AiOutput::Text(t) if t == "hi"));
        match &out[1] {
            AiOutput::ToolCall(tc) => {
                assert_eq!(tc.name, "report");
                let args: Value = serde_json::from_str(&tc.arguments).unwrap();
                assert_eq!(args, json!({"ok": true}));
            }
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    #[test]
    fn parse_anthropic_rejects_max_tokens_empty_and_foreign_shapes() {
        let truncated = json!({
            "stop_reason": "max_tokens",
            "content": [{"type": "tool_use", "name": "report", "input": {}}]
        });
        assert_eq!(
            parse_anthropic_outputs(&truncated).unwrap_err().kind(),
            AiErrorKind::OutputBudget
        );
        let empty = json!({"stop_reason": "end_turn", "content": []});
        assert_eq!(
            parse_anthropic_outputs(&empty).unwrap_err().kind(),
            AiErrorKind::Schema
        );
        let string_input = json!({"stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "report", "input": "{}"}]});
        assert_eq!(
            parse_anthropic_outputs(&string_input).unwrap_err().kind(),
            AiErrorKind::Schema
        );
        // Ответы других протоколов anthropic-парсером не принимаются.
        assert!(parse_anthropic_outputs(&json!({"output": []})).is_err());
        assert!(parse_anthropic_outputs(&json!({"choices": []})).is_err());
    }

    #[test]
    fn parse_chat_text_and_tool_call() {
        let body = json!({"choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": null,
                "tool_calls": [{"function": {"name": "some_tool", "arguments": "{\"a\":1}"}}]
            }
        }]});
        let out = parse_chat_outputs(&body).unwrap();
        assert!(matches!(&out[0], AiOutput::ToolCall(tc) if tc.name == "some_tool"));

        let body = json!({"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]});
        let out = parse_chat_outputs(&body).unwrap();
        assert!(matches!(&out[0], AiOutput::Text(t) if t == "hi"));
    }

    #[test]
    fn parse_chat_rejects_length_empty_and_responses_shape() {
        let length = json!({"choices": [{"finish_reason": "length", "message": {"content": ""}}]});
        assert_eq!(
            parse_chat_outputs(&length).unwrap_err().kind(),
            AiErrorKind::OutputBudget
        );
        let empty = json!({"choices": [{"finish_reason": "stop", "message": {}}]});
        assert_eq!(
            parse_chat_outputs(&empty).unwrap_err().kind(),
            AiErrorKind::Schema
        );
        // Тело Responses API парсером chat не принимается — кросс-протокольный
        // ответ должен падать, а не тихо давать пустой результат.
        let responses_shape = json!({"output": []});
        assert!(parse_chat_outputs(&responses_shape).is_err());
    }
}
