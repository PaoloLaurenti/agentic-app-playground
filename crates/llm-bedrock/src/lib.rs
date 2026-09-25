//! The Amazon Bedrock implementation of [`llm_core::LlmClient`], over the Converse API.

use aws_config::{BehaviorVersion, Region};

/// A Bedrock Runtime client for one region.
///
/// Build it once at startup: loading the AWS configuration and resolving credentials is
/// expensive. Then share it: the SDK client inside is reference counted, so sharing costs nothing.
pub struct BedrockClient {
    client: aws_sdk_bedrockruntime::Client,
}

impl BedrockClient {
    /// Loads the AWS configuration, whose profile and credentials come from the environment and
    /// `~/.aws/config`, and builds a client that calls `region`.
    pub async fn new(region: impl Into<String>) -> Self {
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(region.into()))
            .load()
            .await;
        Self {
            client: aws_sdk_bedrockruntime::Client::new(&config),
        }
    }

    /// The region every call goes to.
    pub fn region(&self) -> Option<&str> {
        self.client.config().region().map(|region| region.as_ref())
    }
}
