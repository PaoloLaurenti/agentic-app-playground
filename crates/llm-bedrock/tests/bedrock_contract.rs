//! Contract tests against the real Bedrock, with the AWS profile and region in `.env`: they check
//! that what `BedrockClient` sends is what Bedrock accepts. They need the network and valid
//! credentials, so they are ignored by default: `cargo nextest run -p llm-bedrock --run-ignored only`.

use llm_bedrock::BedrockClient;
use llm_core::{ContentBlock, LlmClient, LlmRequest, Message, ModelId, Role, StopReason};

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
    };

    let response = client.complete(request).await.unwrap();

    assert_eq!(response.stop_reason, StopReason::EndTurn);
}
