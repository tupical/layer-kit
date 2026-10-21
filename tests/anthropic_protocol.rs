//! E2E-критерий задачи «ApiProtocol::AnthropicMessages для слоёв»: конфиг с
//! `api_protocol=anthropic_messages`, пришедший как `params.ai`, должен
//! ударить в `{голый хост}/v1/messages` с `x-api-key` и `anthropic-version`,
//! без Bearer. Стаб отвечает ТОЛЬКО на этот путь и записывает путь, заголовки
//! и тело, так что ложного зелёного не бывает (образец — chat_protocol.rs).

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::routing::post;
use axum::Json;
use layer_kit::ai::{extract_ai_config, AiErrorKind, AiOutput, AiProvider, AiRequest};
use layer_kit::openai::OpenAiProvider;
use serde_json::{json, Value};

#[derive(Debug)]
struct Seen {
    path: String,
    headers: HeaderMap,
    body: Value,
}

type Log = Arc<Mutex<Vec<Seen>>>;

async fn messages(
    State(log): State<Log>,
    uri: Uri,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    let truncated = body["messages"][0]["content"] == "truncate";
    log.lock().unwrap().push(Seen {
        path: uri.path().to_owned(),
        headers,
        body,
    });
    Json(if truncated {
        json!({
            "stop_reason": "max_tokens",
            "content": [{"type": "tool_use", "name": "report", "input": {}}],
            "usage": {"input_tokens": 3, "output_tokens": 16}
        })
    } else {
        json!({
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "reporting"},
                {"type": "tool_use", "id": "toolu_1", "name": "report", "input": {"ok": true}}
            ],
            "usage": {"input_tokens": 7, "output_tokens": 5}
        })
    })
}

async fn other(State(log): State<Log>, uri: Uri, headers: HeaderMap) -> (StatusCode, &'static str) {
    log.lock().unwrap().push(Seen {
        path: uri.path().to_owned(),
        headers,
        body: Value::Null,
    });
    (StatusCode::NOT_FOUND, "only /v1/messages exists here")
}

#[tokio::test]
async fn anthropic_messages_protocol_reaches_the_messages_endpoint() {
    let log: Log = Log::default();
    let app = axum::Router::new()
        .route("/v1/messages", post(messages))
        .fallback(other)
        .with_state(log.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Конфиг приходит тем же путём, что с платформы: через params.ai; base —
    // голый хост, как у api.anthropic.com.
    let mut params = json!({"ai": {
        "api_key": "sk-ant-k",
        "base_url": format!("http://{addr}"),
        "model": "claude-test",
        "max_output_tokens": 300,
        "api_protocol": "anthropic_messages"
    }});
    let cfg = extract_ai_config(&mut params).expect("anthropic_messages is a valid ai config");
    // no_proxy(): HTTP_PROXY dev-хоста иначе отвечает за loopback-стаб.
    let http = OpenAiProvider::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    let provider = OpenAiProvider::with_http_client(cfg, http);

    let (out, usage) = provider
        .respond_with_usage(AiRequest {
            input: Value::String("ping".into()),
            tools: vec![json!({"type": "function", "name": "report",
                               "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}}})],
            tool_choice: Some("required".into()),
        })
        .await
        .expect("anthropic call against a messages-only stub must succeed");

    assert!(matches!(&out[0], AiOutput::Text(t) if t == "reporting"));
    match &out[1] {
        AiOutput::ToolCall(tc) => {
            assert_eq!(tc.name, "report");
            assert_eq!(
                serde_json::from_str::<Value>(&tc.arguments).unwrap(),
                json!({"ok": true})
            );
        }
        other => panic!("expected a tool call, got {other:?}"),
    }
    let usage = usage.expect("usage parsed");
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(12));

    // Обрезанный по max_tokens ответ — ошибка бюджета, а не half-JSON.
    let err = provider
        .respond(AiRequest {
            input: Value::String("truncate".into()),
            tools: vec![],
            tool_choice: None,
        })
        .await
        .expect_err("max_tokens stop must fail");
    assert_eq!(err.kind(), AiErrorKind::OutputBudget, "{err}");

    let log = log.lock().unwrap();
    assert_eq!(log.len(), 2, "{log:?}");
    let seen = &log[0];
    assert_eq!(seen.path, "/v1/messages");
    assert_eq!(seen.headers["x-api-key"], "sk-ant-k");
    assert_eq!(seen.headers["anthropic-version"], "2023-06-01");
    assert!(
        seen.headers.get("authorization").is_none(),
        "no Bearer for Anthropic"
    );
    assert_eq!(seen.body["model"], "claude-test");
    assert_eq!(seen.body["max_tokens"], 300);
    assert_eq!(
        seen.body["messages"],
        json!([{"role": "user", "content": "ping"}])
    );
    assert_eq!(seen.body["tools"][0]["name"], "report");
    assert!(seen.body["tools"][0].get("input_schema").is_some());
    assert_eq!(seen.body["tool_choice"], json!({"type": "any"}));
    assert_eq!(log[1].path, "/v1/messages");
}

/// Ответ-ошибка Anthropic: `{"type":"error","error":{...}}` с HTTP 401/529.
async fn failing_messages(
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let (status, kind, message) = if headers["x-api-key"] == "sk-ant-revoked" {
        (
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid x-api-key",
        )
    } else {
        assert_eq!(body["messages"][0]["content"], "overload");
        (
            StatusCode::from_u16(529).unwrap(),
            "overloaded_error",
            "Overloaded",
        )
    };
    (
        status,
        Json(json!({"type": "error", "error": {"type": kind, "message": message}})),
    )
}

fn anthropic_provider(addr: std::net::SocketAddr, key: &str) -> OpenAiProvider {
    let mut params = json!({"ai": {
        "api_key": key,
        "base_url": format!("http://{addr}"),
        "model": "claude-test",
        "api_protocol": "anthropic_messages"
    }});
    let cfg = extract_ai_config(&mut params).expect("anthropic_messages is a valid ai config");
    let http = OpenAiProvider::http_client_builder()
        .no_proxy()
        .build()
        .unwrap();
    OpenAiProvider::with_http_client(cfg, http)
}

/// Ошибка провайдера — транспортная (`Other`), а не `Schema`: иначе fujin
/// уходит в repair-ретрай модели на отозванном ключе или перегрузке. Статус и
/// тип ошибки Anthropic видны оператору, ключ в текст ошибки не попадает.
#[tokio::test]
async fn anthropic_error_replies_surface_as_transport_errors_without_the_key() {
    let app = axum::Router::new().route("/v1/messages", post(failing_messages));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let request = |input: &str| AiRequest {
        input: Value::String(input.into()),
        tools: vec![],
        tool_choice: None,
    };

    let err = anthropic_provider(addr, "sk-ant-revoked")
        .respond(request("ping"))
        .await
        .expect_err("401 must fail");
    assert_eq!(err.kind(), AiErrorKind::Other, "{err}");
    let text = err.to_string();
    assert!(text.contains("status 401"), "{text}");
    assert!(text.contains("authentication_error"), "{text}");
    assert!(!text.contains("sk-ant-revoked"), "key leaked: {text}");

    let err = anthropic_provider(addr, "sk-ant-live")
        .respond(request("overload"))
        .await
        .expect_err("529 must fail");
    assert_eq!(err.kind(), AiErrorKind::Other, "{err}");
    let text = err.to_string();
    assert!(text.contains("status 529"), "{text}");
    assert!(text.contains("overloaded_error"), "{text}");
    assert!(!text.contains("sk-ant-live"), "key leaked: {text}");
}

/// Режим без провайдера: нет `ai`-блока, неполный или с неизвестным
/// протоколом блок — конфига нет (слой уходит в детерминированный путь), а
/// секрет всё равно вырезан из аргументов до доменной обработки.
#[test]
fn without_a_usable_anthropic_block_there_is_no_provider_and_no_secret_left() {
    let mut absent = json!({"body": "material"});
    assert!(extract_ai_config(&mut absent).is_none());
    assert_eq!(absent, json!({"body": "material"}));

    for ai in [
        // Опечатка в протоколе — отказ, не тихий откат на Responses.
        json!({"api_key": "sk-ant-k", "base_url": "https://api.anthropic.com",
               "model": "claude-test", "api_protocol": "anthropic"}),
        // Нет ключа.
        json!({"base_url": "https://api.anthropic.com", "model": "claude-test",
               "api_protocol": "anthropic_messages"}),
        // Нет модели.
        json!({"api_key": "sk-ant-k", "base_url": "https://api.anthropic.com",
               "api_protocol": "anthropic_messages"}),
    ] {
        let mut params = json!({"body": "material", "ai": ai.clone()});
        assert!(extract_ai_config(&mut params).is_none(), "{ai}");
        assert_eq!(params, json!({"body": "material"}), "secret left in {ai}");
    }
}
