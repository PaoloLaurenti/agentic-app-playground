//! The prompt registry in Langfuse. The prompts the app runs live in the repository (ADR 0009):
//! Langfuse only receives copies of them, numbers them, and links each version to the generations
//! that used it.

use prompts::Prompt;
use serde_json::{Value, json};

use crate::{Error, LangfuseConfig, basic_authorization};

/// Where prompt versions are kept. `just prompts-push` depends on this trait, not on Langfuse, so
/// its tests can use a fake of their own.
#[async_trait::async_trait]
pub trait PromptRegistry: Send + Sync {
    /// One version of a prompt, or `None` when the registry does not have it.
    async fn version(&self, name: &str, version: u32) -> Result<Option<Prompt>, Error>;

    /// Stores the prompt's messages and config as a new version, and returns its number. The
    /// prompt's own `version` is ignored: the registry chooses the number.
    async fn create(&self, prompt: &Prompt) -> Result<u32, Error>;
}

/// The prompt registry of one Langfuse project, through the public API.
pub struct PromptClient {
    http: reqwest::Client,
    endpoint: String,
    authorization: String,
}

impl PromptClient {
    pub fn new(config: &LangfuseConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: format!(
                "{}/api/public/v2/prompts",
                config.host.trim_end_matches('/')
            ),
            authorization: basic_authorization(config),
        }
    }
}

#[async_trait::async_trait]
impl PromptRegistry for PromptClient {
    async fn version(&self, name: &str, version: u32) -> Result<Option<Prompt>, Error> {
        let response = self
            .http
            .get(format!("{}/{name}?version={version}", self.endpoint))
            .header("Authorization", &self.authorization)
            .send()
            .await?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let reason = response.text().await.unwrap_or_default();
            return Err(Error::PromptRejected { status, reason });
        }
        // A chat prompt comes back with its messages under `prompt`, the same shape as a prompt
        // file's `messages`.
        let mut stored: Value = response.json().await?;
        if let Some(messages) = stored.get_mut("prompt").map(Value::take) {
            stored["messages"] = messages;
        }
        serde_json::from_value(stored)
            .map(Some)
            .map_err(|err| Error::UnexpectedPrompt(err.to_string()))
    }

    async fn create(&self, prompt: &Prompt) -> Result<u32, Error> {
        // No labels: Langfuse marks the new version `latest` by itself, and the app never reads
        // labels (ADR 0009).
        let body = json!({
            "name": prompt.name,
            "type": "chat",
            "prompt": prompt.messages,
            "config": prompt.config,
        });
        let response = self
            .http
            .post(&self.endpoint)
            .header("Authorization", &self.authorization)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let reason = response.text().await.unwrap_or_default();
            return Err(Error::PromptRejected { status, reason });
        }
        let created: Value = response.json().await?;
        created["version"]
            .as_u64()
            .and_then(|version| u32::try_from(version).ok())
            .ok_or_else(|| Error::UnexpectedPrompt(format!("no version in {created}")))
    }
}
