//! Contract tests against the real Bedrock, with the AWS profile and region in `.env`: they check
//! that what `BedrockClient` sends is what Bedrock accepts. They need the network and valid
//! credentials, so they are ignored by default: `cargo nextest run -p llm-bedrock --run-ignored only`.

use llm_bedrock::BedrockClient;
use llm_core::{
    ContentBlock, LlmClient, LlmRequest, Message, ModelId, OutputSchema, Role, StopReason,
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
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text(
                "Answer with a single word: ready?".into(),
            )],
        }],
        max_tokens: 50,
        extra: serde_json::json!({ "output_config": { "effort": "low" } }),
        prompt: None,
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
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Is Rome the capital of Italy?".into())],
        }],
        max_tokens: 100,
        extra: serde_json::Value::Null,
        prompt: None,
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

    let ContentBlock::Text(text) = &response.message.content[0] else {
        panic!("expected text, got {:?}", response.message.content);
    };
    let reply: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        matches!(reply["answer"].as_str(), Some("yes" | "no")),
        "{reply}"
    );
}
