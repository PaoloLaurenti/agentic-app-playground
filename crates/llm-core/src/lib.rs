//! Provider-neutral types for talking to an LLM, and the `LlmClient` trait that every provider
//! implements. This crate must not depend on any provider SDK.

use std::time::Duration;

use futures::stream::BoxStream;

/// The id a provider uses for a model, such as a Bedrock inference profile id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemBlock {
    Text(String),
    /// Marks the end of the stable prefix that the provider may cache.
    CachePoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    Text(String),
    /// Marks the end of the stable prefix that the provider may cache.
    CachePoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmRequest {
    pub model: ModelId,
    pub system: Vec<SystemBlock>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    /// Model-specific fields, such as effort, passed to the provider as they are.
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    /// A reason this crate does not model yet, with the provider's own name for it.
    Other(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_write_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmResponse {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Usage,
    pub latency: Duration,
}

/// An event of a streamed response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmEvent {
    /// A piece of the reply text.
    TextDelta(String),
    /// The model stopped generating.
    Stop(StopReason),
    /// Token usage and latency, sent once the response is complete.
    Metadata { usage: Usage, latency: Duration },
}

/// Errors grouped by what the caller can do about them: retry, fix the request, or fix the
/// configuration. A provider returns them after its own retries have run out.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Too many requests or tokens for the quota. Retryable, later or on another model.
    #[error("throttled: {0}")]
    Throttled(String),
    /// The model is not ready to serve yet. Retryable.
    #[error("model not ready: {0}")]
    ModelNotReady(String),
    /// A transient failure of the provider or of the network. Retryable.
    #[error("service unavailable: {0}")]
    ServiceUnavailable(String),
    /// The request is wrong: a bug to fix, not to retry.
    #[error("invalid request: {0}")]
    ValidationError(String),
    /// Credentials, permissions or model access: a configuration to fix, not to retry.
    #[error("access denied: {0}")]
    AccessDenied(String),
    /// The input does not fit the model's context window: shorten it, do not retry it as is.
    #[error("context too long: {0}")]
    ContextTooLong(String),
    /// Anything else.
    #[error("{0}")]
    Other(String),
}

/// A client for an LLM provider. The harness only ever talks to this trait, never to an SDK.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// The request as this client will actually send it: a provider drops what the target model
    /// does not accept, and a decorator returns what the client inside it would send. There is no
    /// default, so that every new client has to decide.
    fn prepare(&self, req: LlmRequest) -> LlmRequest;

    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError>;

    async fn stream(
        &self,
        req: LlmRequest,
    ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::executor::block_on;

    use super::*;

    /// Answers every request with the same reply, without any provider behind it.
    struct FixedReplyClient;

    #[async_trait::async_trait]
    impl LlmClient for FixedReplyClient {
        fn prepare(&self, req: LlmRequest) -> LlmRequest {
            req
        }

        async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text("Ready.".into())],
                },
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 15,
                    output_tokens: 5,
                    ..Usage::default()
                },
                latency: Duration::from_millis(700),
            })
        }

        async fn stream(
            &self,
            _req: LlmRequest,
        ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
            let events = vec![
                Ok(LlmEvent::TextDelta("Ready.".into())),
                Ok(LlmEvent::Stop(StopReason::EndTurn)),
            ];
            Ok(futures::stream::iter(events).boxed())
        }
    }

    #[test]
    fn a_client_can_be_used_behind_dyn_llm_client() {
        let client: Arc<dyn LlmClient> = Arc::new(FixedReplyClient);
        let request = LlmRequest {
            model: ModelId::new("test-model"),
            system: vec![],
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text(
                    "Answer with a single word: ready?".into(),
                )],
            }],
            max_tokens: 50,
            extra: serde_json::Value::Null,
        };

        let response = block_on(client.complete(request)).unwrap();

        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.output_tokens, 5);
    }
}
