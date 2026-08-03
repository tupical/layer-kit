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

use serde_json::{json, Value};

pub use crate::ai::AiConfig;
use crate::ai::{parse_usage, AiError, AiOutput, AiProvider, AiRequest, AiUsage, ToolCall};
use crate::ai::ApiProtocol;

/// [`AiProvider`] backed by the OpenAI Responses API. Clone is cheap (the
/// inner `reqwest::Client` is Arc-backed).
#[derive(Clone)]
pub struct OpenAiProvider {
    http: reqwest::Client,
    cfg: AiConfig,
}

impl OpenAiProvider {
    pub fn new(cfg: AiConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            cfg,
        }
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
            .map_err(|e| AiError::new(format!("ai response decode failed: {e}")))?;
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
        ApiProtocol::ChatCompletions => (cfg.chat_completions_url(), build_chat_request_body(cfg, req)),
    }
}

/// Default `max_output_tokens` when the config does not set one. Layer
/// operations return short structured tool-calls; a couple thousand tokens is
/// generous while keeping proxy billers' cost forecast (and thus the balance
/// reserve) small.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 2000;

/// Build the Responses API request body. Pure — unit-tested without network.
fn build_request_body(cfg: &AiConfig, req: &AiRequest) -> Value {
    let mut obj = json!({
        "model": cfg.model,
        "input": req.input,
        "max_output_tokens": cfg.max_output_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
    });
    if !req.tools.is_empty() {
        obj["tools"] = Value::Array(req.tools.clone());
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
                    let mut function = serde_json::Map::new();
                    for field in ["name", "description", "parameters"] {
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
        .ok_or_else(|| AiError::new("response missing 'choices[0]'"))?;
    if choice["finish_reason"] == "length" {
        return Err(AiError::new(
            "chat completion exhausted its token budget; increase max_output_tokens",
        ));
    }
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| AiError::new("response missing 'choices[0].message'"))?;
    let mut out = Vec::new();
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        out.push(AiOutput::Text(content.to_owned()));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let function = &tool_call["function"];
            let name = function["name"]
                .as_str()
                .ok_or_else(|| AiError::new("tool call missing function.name"))?;
            let arguments = function["arguments"]
                .as_str()
                .ok_or_else(|| AiError::new("tool call function.arguments must be a JSON string"))?;
            serde_json::from_str::<Value>(arguments).map_err(|error| {
                AiError::new(format!("invalid tool call function.arguments JSON: {error}"))
            })?;
            out.push(AiOutput::ToolCall(ToolCall {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            }));
        }
    }
    if out.is_empty() {
        return Err(AiError::new(
            "chat response has neither content nor tool_calls",
        ));
    }
    Ok(out)
}

/// Parse the `output` array of a Responses API reply into lib [`AiOutput`]s.
/// Pure — unit-tested without network.
fn parse_outputs(body: &Value) -> Result<Vec<AiOutput>, AiError> {
    let items = body["output"]
        .as_array()
        .ok_or_else(|| AiError::new("response missing 'output' array"))?;
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
                out.push(AiOutput::ToolCall(ToolCall {
                    name: item["name"].as_str().unwrap_or("").to_owned(),
                    arguments: item["arguments"].as_str().unwrap_or("{}").to_owned(),
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

    fn cfg(max_output_tokens: Option<u32>) -> AiConfig {
        AiConfig {
            api_key: "sk-test".into(),
            base_url: "https://api.example.com/v1".into(),
            model: "gpt-4.1".into(),
            max_output_tokens,
            api_protocol: ApiProtocol::default(),
        }
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
        assert!(parse_outputs(&json!({"id": "resp_1"})).is_err());
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
        assert!(parse_chat_outputs(&length).is_err());
        let empty = json!({"choices": [{"finish_reason": "stop", "message": {}}]});
        assert!(parse_chat_outputs(&empty).is_err());
        // Тело Responses API парсером chat не принимается — кросс-протокольный
        // ответ должен падать, а не тихо давать пустой результат.
        let responses_shape = json!({"output": []});
        assert!(parse_chat_outputs(&responses_shape).is_err());
    }
}
