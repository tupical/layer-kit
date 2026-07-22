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
        let body = build_request_body(&self.cfg, &req);
        let resp = self
            .http
            .post(self.cfg.responses_url())
            .bearer_auth(&self.cfg.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::new(format!("responses request failed: {e}")))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let message = resp.text().await.unwrap_or_default();
            return Err(AiError::new(format!(
                "responses api status {status}: {message}"
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| AiError::new(format!("responses decode failed: {e}")))?;
        Ok((parse_outputs(&body)?, parse_usage(&body)))
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
}
