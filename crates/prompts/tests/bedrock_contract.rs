//! Contract tests against the real Bedrock, with the AWS profile, region and models in `.env`:
//! they check that Bedrock accepts the schemas this crate generates. They need the network and
//! valid credentials, so they are ignored by default:
//! `cargo nextest run -p prompts --run-ignored only`.

use llm_bedrock::BedrockClient;
use llm_core::{ContentBlock, LlmClient, LlmRequest, ModelId};
use prompts::{SafetyVerdict, safety_classifier};

#[tokio::test]
#[ignore = "talks to the real Bedrock with the AWS profile in .env"]
async fn the_safety_classifier_replies_with_a_verdict_that_follows_its_schema() {
    // `.env` lives at the repository root; dotenvy looks for it in the parent folders too.
    let _ = dotenvy::dotenv();
    let region = std::env::var("AWS_REGION").expect("AWS_REGION must be set in .env");
    let model =
        std::env::var("BEDROCK_MODEL_FAST").expect("BEDROCK_MODEL_FAST must be set in .env");
    let client = BedrockClient::new(region).await;
    let rendered = safety_classifier()
        .unwrap()
        .render(&[("message", "Vorrei prenotare una visita dermatologica.")])
        .unwrap();
    let request = LlmRequest {
        model: ModelId::new(model),
        system: rendered.system,
        messages: rendered.messages,
        max_tokens: 200,
        extra: serde_json::Value::Null,
        prompt: Some(rendered.prompt),
        output_schema: Some(SafetyVerdict::output_schema()),
    };

    let response = client.complete(request).await.unwrap();

    let ContentBlock::Text(text) = &response.message.content[0] else {
        panic!("expected text, got {:?}", response.message.content);
    };
    let verdict: Result<SafetyVerdict, _> = serde_json::from_str(text);
    assert!(verdict.is_ok(), "{text}");
}
