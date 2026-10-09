//! Contract tests against the real Langfuse project in `.env`: they check that what `ScoreClient`
//! and `PromptClient` send is what Langfuse accepts and stores, and that every prompt file names a
//! version whose text Langfuse holds. They need the network and the keys, so they are
//! ignored by default: `cargo nextest run -p observability --run-ignored only`.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use observability::{LangfuseConfig, PromptClient, PromptRegistry, ScoreClient, Scores};
use prompts::Prompt;
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

    async fn delete_prompt_version(&self, name: &str, version: u32) {
        let response = self
            .http
            .delete(format!(
                "{}/api/public/v2/prompts/{name}?version={version}",
                self.host
            ))
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

/// The ingestion API answers `207` when it rejects an event, with the reason in the body, so a
/// rejected score is not a failed request (ADR 0010).
#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn a_score_the_ingestion_rejects_is_an_error() {
    let config = langfuse_from_env();
    let scores: &dyn Scores = &ScoreClient::new(&config);

    let err = scores
        .boolean(&unused_trace_id(), "", true)
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("400"), "{err}");
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

/// Langfuse Cloud's Hobby plan allows 30 requests a minute to its general API, shared by the whole
/// organization (ADR 0010). The scores are filed under an environment of their own and not
/// deleted, since deleting them one by one would run into the same limit.
#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn a_burst_of_scores_above_the_general_api_limit_is_all_accepted() {
    let config = LangfuseConfig {
        environment: "contract-test".into(),
        ..langfuse_from_env()
    };
    let scores: &dyn Scores = &ScoreClient::new(&config);

    let mut rejected = vec![];
    for _ in 0..35 {
        if let Err(err) = scores
            .boolean(&unused_trace_id(), "contract_test_burst", true)
            .await
        {
            rejected.push(err.to_string());
        }
    }

    assert!(
        rejected.is_empty(),
        "{} rejected: {rejected:?}",
        rejected.len()
    );
}

/// Every prompt in the repository's `prompts/` folder.
fn repository_prompts() -> Vec<Prompt> {
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../prompts");
    std::fs::read_dir(&folder)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "toml")
        })
        .map(|path| Prompt::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap())
        .collect()
}

#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn every_prompt_file_has_the_text_of_the_version_it_names() {
    let registry: &dyn PromptRegistry = &PromptClient::new(&langfuse_from_env());
    let files = repository_prompts();
    assert!(!files.is_empty(), "no prompt files found");

    for file in files {
        let stored = registry
            .version(&file.name, file.version)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("{} v{} is not in Langfuse", file.name, file.version));
        assert_eq!(stored, file);
    }
}

#[tokio::test]
#[ignore = "talks to the real Langfuse project in .env"]
async fn a_created_version_reads_back_with_its_messages_and_config() {
    let config = langfuse_from_env();
    let registry: &dyn PromptRegistry = &PromptClient::new(&config);
    // A text that no earlier run used, so that Langfuse stores a new version.
    let prompt = Prompt::from_toml(&format!(
        r#"
name = "contract-test"
version = 0

[config]
max_tokens = 10

[[messages]]
role = "system"
content = "Run {}."

[[messages]]
role = "user"
content = "<message>{{{{message}}}}</message>"
"#,
        unused_trace_id()
    ))
    .unwrap();

    let version = registry.create(&prompt).await.unwrap();
    let stored = registry.version(&prompt.name, version).await;
    LangfuseApi::new(&config)
        .delete_prompt_version(&prompt.name, version)
        .await;

    let stored = stored.unwrap().expect("the new version is not in Langfuse");
    assert_eq!(stored, Prompt { version, ..prompt });
}
