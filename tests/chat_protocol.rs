//! E2E-критерий задачи «Читать ai.api_protocol на стороне сервисов слоёв»:
//! конфиг с `api_protocol=chat_completions`, пришедший как `params.ai`,
//! должен ударить именно в `/chat/completions`. Стаб отвечает ТОЛЬКО на этот
//! путь и записывает все запрошенные пути, так что ложного зелёного не
//! бывает (образец — cabinet_routes::stored_chat_completions_protocol_
//! reaches_the_outgoing_request в mcpbox.ru).

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::routing::post;
use axum::Json;
use layer_kit::ai::{extract_ai_config, AiOutput, AiProvider, AiRequest};
use layer_kit::openai::OpenAiProvider;
use serde_json::{json, Value};

type Seen = Arc<Mutex<Vec<String>>>;

async fn chat(State(seen): State<Seen>, uri: Uri) -> Json<Value> {
    seen.lock().unwrap().push(uri.path().to_owned());
    Json(json!({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": null,
                "tool_calls": [{"function": {"name": "report", "arguments": "{\"ok\":true}"}}]
            }
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
}

async fn other(State(seen): State<Seen>, uri: Uri) -> (StatusCode, &'static str) {
    seen.lock().unwrap().push(uri.path().to_owned());
    (StatusCode::NOT_FOUND, "only /chat/completions exists here")
}

#[tokio::test]
async fn chat_completions_protocol_reaches_the_chat_endpoint() {
    let seen: Seen = Seen::default();
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(chat))
        .fallback(other)
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Конфиг приходит тем же путём, что с платформы: через params.ai.
    let mut params = json!({"ai": {
        "api_key": "k",
        "base_url": format!("http://{addr}/v1"),
        "model": "m",
        "api_protocol": "chat_completions"
    }});
    let cfg = extract_ai_config(&mut params).expect("valid ai config");
    let provider = OpenAiProvider::new(cfg);

    let out = provider
        .respond(AiRequest {
            input: Value::String("ping".into()),
            tools: vec![json!({"type": "function", "name": "report",
                               "parameters": {"type": "object", "properties": {}}})],
            tool_choice: Some("required".into()),
        })
        .await
        .expect("chat-protocol call against a chat-only stub must succeed");

    assert!(matches!(&out[0], AiOutput::ToolCall(tc) if tc.name == "report"));
    assert_eq!(seen.lock().unwrap().as_slice(), ["/v1/chat/completions"]);
}
