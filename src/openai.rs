//! Concrete AI provider for a layer server: a minimal OpenAI Responses API
//! client implementing the [`crate::ai::AiProvider`] seam.
//!
//! Config (env), mirroring the daruma ai-infra contract so one key serves
//! the whole co-located platform:
//!   OPENAI_API_KEY  — required; when unset/empty `AiConfig::from_env`
//!                     returns `None` and AI methods answer `ai_not_configured`
//!                     instead of failing at call time.
//!   OPENAI_BASE_URL — default `https://api.openai.com/v1`.
//!   OPENAI_MODEL    — default `gpt-4.1`.
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
        let (url, body) = endpoint_and_body(&self.cfg, &req);
        let resp = self
            .http
            .post(url)
            .bearer_auth(&self.cfg.api_key)
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
            .map_err(|e| AiError::schema(format!("ai response decode failed: {e}")))?;
        let outputs = match self.cfg.api_protocol {
            ApiProtocol::Responses => parse_outputs(&body)?,
            ApiProtocol::ChatCompletions => parse_chat_outputs(&body)?,
        };
        Ok((outputs, parse_usage(&body)))
    }
}

/// Endpoint + matching body for the configured protocol. Split out so a unit
/// test can prove the pairing without a network: crossed arms here (right URL,
/// wrong body) are exactly the outage where every layer posted Responses
/// bodies to a Chat-Completions-only provider.
fn endpoint_and_body(cfg: &AiConfig, req: &AiRequest) -> (String, Value) {
    match cfg.api_protocol {
        ApiProtocol::Responses => (cfg.responses_url(), build_request_body(cfg, req)),
        ApiProtocol::ChatCompletions => (
            cfg.chat_completions_url(),
            build_chat_request_body(cfg, req),
        ),
    }
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

        let (url, body) = endpoint_and_body(&cfg(None), &req);
        assert!(url.ends_with("/responses"), "{url}");
        assert_eq!(body["input"], "p");
        assert_eq!(body["tools"][0]["name"], "some_tool");

        let mut chat_cfg = cfg(Some(512));
        chat_cfg.api_protocol = ApiProtocol::ChatCompletions;
        let (url, body) = endpoint_and_body(&chat_cfg, &req);
        assert!(url.ends_with("/chat/completions"), "{url}");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "p");
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "some_tool");
        assert!(body.get("input").is_none());
        assert!(body.get("max_output_tokens").is_none());
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
