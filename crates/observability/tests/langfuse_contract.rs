//! Contract tests against the real Langfuse project in `.env`: they check that what `ScoreClient`
//! sends is what Langfuse accepts and stores. They need the network and the keys, so they are
//! ignored by default: `cargo nextest run -p observability --run-ignored only`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use observability::{LangfuseConfig, ScoreClient, Scores};
use serde_json::Value;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set in .env"))
}

fn langfuse_from_env() -> LangfuseConfig {
    // `.env` lives at the repository root; dotenvy looks for it in the parent folders too.
    let _ = dotenvy::dotenv();
    LangfuseConfig {
        host: env("LANGFUSE_HOST"),
        public_key: env("LANGFUSE_PUBLIC_KEY"),
        secret_key: env("LANGFUSE_SECRET_KEY"),
        environment: env("APP_ENVIRONMENT"),
        release: "contract-test".into(),
    }
}

/// A trace id that no real trace uses: 32 hex digits from the current time.
fn unused_trace_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{nanos:032x}")
}

/// The public API with plain HTTP, independent of the code under test: reads a score back, and
/// deletes it at the end.
struct LangfuseApi {
    http: reqwest::Client,
    host: String,
    authorization: String,
}

impl LangfuseApi {
    fn new(config: &LangfuseConfig) -> Self {
        let credentials = BASE64.encode(format!("{}:{}", config.public_key, config.secret_key));
        Self {
            http: reqwest::Client::new(),
            host: config.host.clone(),
            authorization: format!("Basic {credentials}"),
        }
    }

    /// Scores are stored asynchronously: in the tests of 2026-10-06 a new score took between 20 and
    /// 60 seconds to become readable, so the score is polled for up to two minutes.
    async fn score(&self, id: &str) -> Option<Value> {
        for _ in 0..60 {
            let page: Value = self
                .http
                .get(format!("{}/api/public/v3/scores?id={id}", self.host))
                .header("Authorization", &self.authorization)
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if let Some(score) = page["data"].as_array().and_then(|data| data.first()) {
                return Some(score.clone());
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        None
    }

    async fn delete_score(&self, id: &str) {
        let response = self
            .http
            .delete(format!("{}/api/public/scores/{id}", self.host))
            .header("Authorization", &self.authorization)
            .send()
            .await
            .unwrap();
        let status = response.status();
        assert!(
            status.is_success(),
            "delete: {status} {}",
            response.text().await.unwrap()
        );
    }
}

#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn a_boolean_score_is_stored_with_its_name_value_and_data_type() {
    let config = langfuse_from_env();
    let scores: &dyn Scores = &ScoreClient::new(&config);

    let id = scores
        .boolean(&unused_trace_id(), "contract_test", true)
        .await
        .unwrap();

    let api = LangfuseApi::new(&config);
    let stored = api.score(&id).await;
    api.delete_score(&id).await;
    let stored = stored.expect("the score never showed up in Langfuse");
    assert_eq!(stored["name"], "contract_test");
    assert_eq!(stored["value"], true);
    assert_eq!(stored["dataType"], "BOOLEAN");
    assert_eq!(
        stored["environment"].as_str(),
        Some(config.environment.as_str())
    );
}

#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn a_rejected_score_is_an_error_with_the_status() {
    let config = LangfuseConfig {
        public_key: "pk-lf-not-a-key".into(),
        secret_key: "sk-lf-not-a-key".into(),
        ..langfuse_from_env()
    };
    let scores: &dyn Scores = &ScoreClient::new(&config);

    let err = scores
        .boolean(&unused_trace_id(), "contract_test", true)
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("401"), "{err}");
}

#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn false_is_stored_as_false() {
    let config = langfuse_from_env();
    let scores: &dyn Scores = &ScoreClient::new(&config);

    let id = scores
        .boolean(&unused_trace_id(), "contract_test", false)
        .await
        .unwrap();

    let api = LangfuseApi::new(&config);
    let stored = api.score(&id).await;
    api.delete_score(&id).await;
    assert_eq!(
        stored.expect("the score never showed up in Langfuse")["value"],
        false
    );
}
