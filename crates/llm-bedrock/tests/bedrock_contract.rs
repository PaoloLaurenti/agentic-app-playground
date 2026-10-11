//! Contract tests against the real Bedrock, with the AWS profile and region in `.env`: they check
//! that what `BedrockClient` sends is what Bedrock accepts. They need the network and valid
//! credentials, so they are ignored by default: `cargo nextest run -p llm-bedrock --run-ignored only`.

use llm_bedrock::BedrockClient;
use llm_core::{
    AssistantBlock, InputSchema, LlmClient, LlmRequest, LlmResponse, Message, ModelId,
    OutputSchema, StopReason, ToolDescription, ToolName, ToolOutcome, ToolSpec, ToolUseId,
    UserBlock,
};

const HAIKU_4_5: &str = "eu.anthropic.claude-haiku-4-5-20251001-v1:0";

async fn client_from_env() -> BedrockClient {
    // `.env` lives at the repository root; dotenvy looks for it in the parent folders too.
    let _ = dotenvy::dotenv();
    let region = std::env::var("AWS_REGION").expect("AWS_REGION must be set in .env");
    BedrockClient::new(region).await
}

#[tokio::test]
#[ignore = "talks to the real Bedrock with the AWS profile in .env"]
async fn effort_sent_to_a_model_that_rejects_it_does_not_fail_the_call() {
    let client = client_from_env().await;
    let request = LlmRequest {
        model: ModelId::new(HAIKU_4_5),
        system: vec![],
        messages: vec![Message::User(vec![UserBlock::Text(
            "Answer with a single word: ready?".into(),
        )])],
        max_tokens: 50,
        extra: serde_json::json!({ "output_config": { "effort": "low" } }),
        prompt: None,
        tools: vec![],
        output_schema: None,
    };

    let response = client.complete(request).await.unwrap();

    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
#[ignore = "talks to the real Bedrock with the AWS profile in .env"]
async fn a_model_with_an_output_schema_replies_with_json_that_follows_it() {
    let client = client_from_env().await;
    let request = LlmRequest {
        model: ModelId::new(HAIKU_4_5),
        system: vec![],
        messages: vec![Message::User(vec![UserBlock::Text(
            "Is Rome the capital of Italy?".into(),
        )])],
        max_tokens: 100,
        extra: serde_json::Value::Null,
        prompt: None,
        tools: vec![],
        output_schema: Some(OutputSchema {
            name: "Answer".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": { "answer": { "type": "string", "enum": ["yes", "no"] } },
                "required": ["answer"],
                "additionalProperties": false,
            }),
        }),
    };

    let response = client.complete(request).await.unwrap();

    let AssistantBlock::Text(text) = &response.content[0] else {
        panic!("expected text, got {:?}", response.content);
    };
    let reply: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        matches!(reply["answer"].as_str(), Some("yes" | "no")),
        "{reply}"
    );
}

fn date_tool() -> ToolSpec {
    ToolSpec {
        name: ToolName::new("get_current_date").unwrap(),
        description: ToolDescription::new(
            "Today's date in a time zone. Use it whenever the answer depends on the current date.",
        )
        .unwrap(),
        input_schema: InputSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "time_zone": { "type": "string", "description": "An IANA time zone, such as Europe/Rome." },
            },
            "required": ["time_zone"],
        }))
        .unwrap(),
    }
}

fn date_request(messages: Vec<Message>) -> LlmRequest {
    LlmRequest {
        model: ModelId::new(HAIKU_4_5),
        system: vec![],
        messages,
        max_tokens: 300,
        extra: serde_json::Value::Null,
        prompt: None,
        tools: vec![date_tool()],
        output_schema: None,
    }
}

fn date_question() -> Message {
    Message::User(vec![UserBlock::Text(
        "What is the date today in Rome?".into(),
    )])
}

/// The id, name and input of the first tool request of a reply.
fn tool_use_of(
    response: &LlmResponse,
) -> (
    &ToolUseId,
    &ToolName,
    &serde_json::Map<String, serde_json::Value>,
) {
    response
        .content
        .iter()
        .find_map(|block| match block {
            AssistantBlock::ToolUse { id, name, input } => Some((id, name, input)),
            AssistantBlock::Text(_) => None,
        })
        .unwrap_or_else(|| panic!("expected a tool request, got {:?}", response.content))
}

#[tokio::test]
#[ignore = "talks to the real Bedrock with the AWS profile in .env"]
async fn a_model_given_a_tool_asks_for_it_with_input_that_follows_its_schema() {
    let client = client_from_env().await;

    let response = client
        .complete(date_request(vec![date_question()]))
        .await
        .unwrap();

    assert_eq!(response.stop_reason, StopReason::ToolUse);
    let (_, name, input) = tool_use_of(&response);
    assert_eq!(name.as_str(), "get_current_date");
    assert!(input["time_zone"].is_string(), "{input:?}");
}

#[tokio::test]
#[ignore = "talks to the real Bedrock with the AWS profile in .env"]
async fn a_model_answers_from_the_tool_result_it_receives() {
    let client = client_from_env().await;
    let first = client
        .complete(date_request(vec![date_question()]))
        .await
        .unwrap();
    let (id, _, _) = tool_use_of(&first);
    let id = id.clone();

    let response = client
        .complete(date_request(vec![
            date_question(),
            Message::Assistant(first.content),
            Message::User(vec![UserBlock::ToolResult {
                tool_use_id: id,
                outcome: ToolOutcome::Success("2026-10-12".into()),
            }]),
        ]))
        .await
        .unwrap();

    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(
        response
            .content
            .iter()
            .any(|block| matches!(block, AssistantBlock::Text(text) if !text.trim().is_empty())),
        "{:?}",
        response.content
    );
}
