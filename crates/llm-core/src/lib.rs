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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemBlock {
    Text(String),
    /// Marks the end of the stable prefix that the provider may cache.
    CachePoint,
}

/// A turn of the conversation. Each role has its own kind of block, so that a block can only
/// appear in a turn that may carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    User(Vec<UserBlock>),
    Assistant(Vec<AssistantBlock>),
}

/// What the user's turn may carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserBlock {
    Text(String),
    /// Marks the end of the stable prefix that the provider may cache. Only the user's turns carry
    /// it: the prefix ends with the last message sent, which is always the user's.
    CachePoint,
    /// What came of a tool the model asked for, answering its request by id.
    ToolResult {
        tool_use_id: ToolUseId,
        outcome: ToolOutcome,
    },
}

/// What the model's turn may carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantBlock {
    Text(String),
    /// The model asks the harness to run a tool. The input is a JSON object, since its shape
    /// depends on the tool: the tool itself reads it into its own type.
    ToolUse {
        id: ToolUseId,
        name: ToolName,
        input: serde_json::Map<String, serde_json::Value>,
    },
}

/// The id the model gave a tool request, copied into the result that answers it. The provider
/// creates it, so it is not checked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolUseId(String);

impl ToolUseId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// What a tool sent back to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOutcome {
    /// The tool ran: its result, as text.
    Success(String),
    /// The tool did not run or failed: the reason, written so the model can correct its request.
    Failure(String),
}

/// The name the model uses to ask for a tool: 1 to 64 letters, digits, `_` or `-`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolName(String);

impl ToolName {
    pub fn new(name: impl Into<String>) -> Result<Self, InvalidToolName> {
        let name = name.into();
        let allowed = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
        if (1..=64).contains(&name.len()) && name.chars().all(allowed) {
            Ok(Self(name))
        } else {
            Err(InvalidToolName(name))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
#[error("a tool name must be 1 to 64 letters, digits, `_` or `-`, not {0:?}")]
pub struct InvalidToolName(String);

/// What a tool does, written for the model: when to use it, and when not to. Never blank, or the
/// model could not tell when to ask for the tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDescription(String);

impl ToolDescription {
    pub fn new(description: impl Into<String>) -> Result<Self, BlankToolDescription> {
        let description = description.into();
        if description.trim().is_empty() {
            Err(BlankToolDescription)
        } else {
            Ok(Self(description))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
#[error("a tool's description must not be blank")]
pub struct BlankToolDescription;

/// The JSON Schema of a tool's input: an object whose root has `"type": "object"`, since the
/// input is always a JSON object.
#[derive(Debug, Clone, PartialEq)]
pub struct InputSchema(serde_json::Map<String, serde_json::Value>);

impl InputSchema {
    pub fn new(schema: serde_json::Value) -> Result<Self, InvalidInputSchema> {
        match schema {
            serde_json::Value::Object(schema) if schema.get("type") == Some(&"object".into()) => {
                Ok(Self(schema))
            }
            other => Err(InvalidInputSchema(other)),
        }
    }

    pub fn as_object(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
#[error("a tool's input schema must be an object with \"type\": \"object\", not {0}")]
pub struct InvalidInputSchema(serde_json::Value);

/// A tool as the model is told about it: its name, what it does, and the input it takes.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: ToolName,
    pub description: ToolDescription,
    pub input_schema: InputSchema,
}

/// The prompt a request was built from, so that the call can be linked to that prompt's version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptRef {
    pub name: String,
    pub version: u32,
}

/// A JSON Schema the reply must follow, so that it can be parsed rather than read.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSchema {
    /// A short name for the schema, such as the Rust type it was generated from.
    pub name: String,
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlmRequest {
    pub model: ModelId,
    pub system: Vec<SystemBlock>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    /// Model-specific fields, such as effort, passed to the provider as they are.
    pub extra: serde_json::Value,
    /// The prompt the request was built from, when there is one. The provider ignores it.
    pub prompt: Option<PromptRef>,
    /// The tools the model may ask for: none when empty.
    pub tools: Vec<ToolSpec>,
    /// The schema the reply must follow, when the caller wants structured output.
    pub output_schema: Option<OutputSchema>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    /// The model asks for one or more tools, and waits for their results.
    ToolUse,
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
    /// The model's turn. A reply is always the assistant's, so it carries no role.
    pub content: Vec<AssistantBlock>,
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
                content: vec![AssistantBlock::Text("Ready.".into())],
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
            messages: vec![Message::User(vec![UserBlock::Text(
                "Answer with a single word: ready?".into(),
            )])],
            max_tokens: 50,
            extra: serde_json::Value::Null,
            prompt: None,
            tools: vec![],
            output_schema: None,
        };

        let response = block_on(client.complete(request)).unwrap();

        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn a_tool_name_of_letters_digits_underscores_and_dashes_is_accepted() {
        for name in ["get_current_date", "search-kb2", &"a".repeat(64)] {
            assert_eq!(ToolName::new(name).unwrap().as_str(), name);
        }
    }

    #[test]
    fn a_tool_name_converse_would_reject_is_an_error() {
        for name in ["", &"a".repeat(65), "get date", "get.date"] {
            assert!(ToolName::new(name).is_err(), "{name:?} was accepted");
        }
    }

    #[test]
    fn a_description_with_text_is_kept_as_it_is() {
        let text = "Today's date in the clinic's time zone.";

        assert_eq!(ToolDescription::new(text).unwrap().as_str(), text);
    }

    #[test]
    fn a_blank_description_is_an_error() {
        for description in ["", " \n\t"] {
            assert!(
                ToolDescription::new(description).is_err(),
                "{description:?} was accepted"
            );
        }
    }

    #[test]
    fn an_object_schema_is_an_input_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "date": { "type": "string", "format": "date" } },
            "required": ["date"],
        });

        let input_schema = InputSchema::new(schema.clone()).unwrap();

        assert_eq!(
            serde_json::Value::Object(input_schema.as_object().clone()),
            schema
        );
    }

    #[test]
    fn a_schema_whose_root_is_not_an_object_type_is_an_error() {
        for schema in [
            serde_json::json!([{ "type": "object" }]),
            serde_json::json!(true),
            serde_json::json!({ "properties": {} }),
            serde_json::json!({ "type": "string" }),
        ] {
            assert!(
                InputSchema::new(schema.clone()).is_err(),
                "{schema} was accepted"
            );
        }
    }
}
