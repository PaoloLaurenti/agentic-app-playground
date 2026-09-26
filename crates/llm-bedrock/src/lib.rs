//! The Amazon Bedrock implementation of [`llm_core::LlmClient`], over the Converse API.

use std::fmt::Display;
use std::time::Duration;

use aws_config::retry::RetryConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::{DisplayErrorContext, SdkError};
use aws_sdk_bedrockruntime::operation::converse::{ConverseError, ConverseOutput};
use aws_sdk_bedrockruntime::types as sdk;
use futures::stream::BoxStream;
use llm_core::{
    ContentBlock, LlmClient, LlmError, LlmEvent, LlmRequest, LlmResponse, Message, Role,
    StopReason, SystemBlock, Usage,
};

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
            // The SDK retries throttling and transient errors with exponential backoff and
            // jitter. These are its standard values, written out so that they are visible.
            .retry_config(
                RetryConfig::standard()
                    .with_max_attempts(3)
                    .with_initial_backoff(Duration::from_secs(1))
                    .with_max_backoff(Duration::from_secs(20)),
            )
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

#[async_trait::async_trait]
impl LlmClient for BedrockClient {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let max_tokens = i32::try_from(req.max_tokens).map_err(invalid)?;
        let system = to_sdk_system(&req.system)?;
        let messages = req
            .messages
            .iter()
            .map(to_sdk_message)
            .collect::<Result<Vec<_>, _>>()?;

        let output = self
            .client
            .converse()
            .model_id(req.model.as_str())
            .set_system((!system.is_empty()).then_some(system))
            .set_messages(Some(messages))
            .inference_config(
                sdk::InferenceConfiguration::builder()
                    .max_tokens(max_tokens)
                    .build(),
            )
            .send()
            .await
            .map_err(from_sdk_error)?;

        from_sdk_output(&output)
    }

    async fn stream(
        &self,
        _req: LlmRequest,
    ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
        Err(LlmError::Other("streaming is not implemented yet".into()))
    }
}

/// A request the SDK refuses to build is a bug in the request, not something to retry.
fn invalid(err: impl Display) -> LlmError {
    LlmError::ValidationError(err.to_string())
}

/// Maps an error that is left after the SDK's own retries.
fn from_sdk_error(err: SdkError<ConverseError>) -> LlmError {
    match err {
        SdkError::ServiceError(context) => from_converse_error(context.into_err()),
        other => {
            let message = DisplayErrorContext(&other).to_string();
            match &other {
                SdkError::TimeoutError(_) => LlmError::ServiceUnavailable(message),
                SdkError::DispatchFailure(failure) if failure.is_io() || failure.is_timeout() => {
                    LlmError::ServiceUnavailable(message)
                }
                _ => LlmError::Other(message),
            }
        }
    }
}

/// Maps an exception returned by Bedrock. The `Display` of an exception is its name followed by
/// its message, which is short enough for logs.
fn from_converse_error(err: ConverseError) -> LlmError {
    let message = err.to_string();
    match err {
        ConverseError::ThrottlingException(_) => LlmError::Throttled(message),
        ConverseError::ModelNotReadyException(_) => LlmError::ModelNotReady(message),
        ConverseError::ServiceUnavailableException(_)
        | ConverseError::InternalServerException(_)
        | ConverseError::ModelTimeoutException(_) => LlmError::ServiceUnavailable(message),
        ConverseError::AccessDeniedException(_) => LlmError::AccessDenied(message),
        ConverseError::ValidationException(ref exception)
            if exception.message().is_some_and(is_context_too_long) =>
        {
            LlmError::ContextTooLong(message)
        }
        ConverseError::ValidationException(_) | ConverseError::ResourceNotFoundException(_) => {
            LlmError::ValidationError(message)
        }
        _ => LlmError::Other(message),
    }
}

/// Bedrock reports an input that does not fit the context window as a plain validation error:
/// only its message tells the two apart.
fn is_context_too_long(message: &str) -> bool {
    message.to_lowercase().contains("too long")
}

fn cache_point() -> Result<sdk::CachePointBlock, LlmError> {
    sdk::CachePointBlock::builder()
        .r#type(sdk::CachePointType::Default)
        .build()
        .map_err(invalid)
}

fn to_sdk_system(blocks: &[SystemBlock]) -> Result<Vec<sdk::SystemContentBlock>, LlmError> {
    blocks
        .iter()
        .map(|block| match block {
            SystemBlock::Text(text) => Ok(sdk::SystemContentBlock::Text(text.clone())),
            SystemBlock::CachePoint => cache_point().map(sdk::SystemContentBlock::CachePoint),
        })
        .collect()
}

fn to_sdk_message(message: &Message) -> Result<sdk::Message, LlmError> {
    let role = match message.role {
        Role::User => sdk::ConversationRole::User,
        Role::Assistant => sdk::ConversationRole::Assistant,
    };
    let content = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => Ok(sdk::ContentBlock::Text(text.clone())),
            ContentBlock::CachePoint => cache_point().map(sdk::ContentBlock::CachePoint),
        })
        .collect::<Result<Vec<_>, _>>()?;
    sdk::Message::builder()
        .role(role)
        .set_content(Some(content))
        .build()
        .map_err(invalid)
}

fn from_sdk_output(output: &ConverseOutput) -> Result<LlmResponse, LlmError> {
    let Some(sdk::ConverseOutput::Message(message)) = output.output() else {
        return Err(LlmError::Other("the response contains no message".into()));
    };
    let usage = output
        .usage()
        .ok_or_else(|| LlmError::Other("the response contains no token usage".into()))?;
    let metrics = output
        .metrics()
        .ok_or_else(|| LlmError::Other("the response contains no metrics".into()))?;
    let latency_ms = u64::try_from(metrics.latency_ms())
        .map_err(|_| LlmError::Other(format!("negative latency {}", metrics.latency_ms())))?;

    Ok(LlmResponse {
        message: from_sdk_message(message)?,
        stop_reason: from_sdk_stop_reason(output.stop_reason()),
        usage: Usage {
            input_tokens: token_count(usage.input_tokens())?,
            output_tokens: token_count(usage.output_tokens())?,
            cache_read_tokens: token_count(usage.cache_read_input_tokens().unwrap_or(0))?,
            cache_write_tokens: token_count(usage.cache_write_input_tokens().unwrap_or(0))?,
        },
        latency: Duration::from_millis(latency_ms),
    })
}

fn from_sdk_message(message: &sdk::Message) -> Result<Message, LlmError> {
    let role = match message.role() {
        sdk::ConversationRole::User => Role::User,
        sdk::ConversationRole::Assistant => Role::Assistant,
        other => {
            return Err(LlmError::Other(format!(
                "unexpected role {}",
                other.as_str()
            )));
        }
    };
    // Only text is kept for now: reasoning blocks, which Claude 5 models emit by default, and
    // every other kind of block are skipped.
    let content = message
        .content()
        .iter()
        .filter_map(|block| match block {
            sdk::ContentBlock::Text(text) => Some(ContentBlock::Text(text.clone())),
            _ => None,
        })
        .collect();
    Ok(Message { role, content })
}

fn from_sdk_stop_reason(reason: &sdk::StopReason) -> StopReason {
    match reason {
        sdk::StopReason::EndTurn => StopReason::EndTurn,
        sdk::StopReason::MaxTokens => StopReason::MaxTokens,
        sdk::StopReason::StopSequence => StopReason::StopSequence,
        other => StopReason::Other(other.as_str().to_owned()),
    }
}

fn token_count(count: i32) -> Result<u32, LlmError> {
    u32::try_from(count).map_err(|_| LlmError::Other(format!("negative token count {count}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_keeps_its_role_and_blocks_in_the_sdk() {
        let message = Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text("ready?".into()),
                ContentBlock::CachePoint,
            ],
        };

        let sdk_message = to_sdk_message(&message).unwrap();

        assert_eq!(sdk_message.role(), &sdk::ConversationRole::User);
        assert!(
            matches!(&sdk_message.content()[0], sdk::ContentBlock::Text(text) if text == "ready?")
        );
        assert!(matches!(
            &sdk_message.content()[1],
            sdk::ContentBlock::CachePoint(_)
        ));
    }

    #[test]
    fn a_converse_output_becomes_a_response_without_reasoning_blocks() {
        let reasoning = sdk::ReasoningTextBlock::builder()
            .text("One word is enough.")
            .build()
            .unwrap();
        let message = sdk::Message::builder()
            .role(sdk::ConversationRole::Assistant)
            .content(sdk::ContentBlock::ReasoningContent(
                sdk::ReasoningContentBlock::ReasoningText(reasoning),
            ))
            .content(sdk::ContentBlock::Text("Ready.".into()))
            .build()
            .unwrap();
        let output = ConverseOutput::builder()
            .output(sdk::ConverseOutput::Message(message))
            .stop_reason(sdk::StopReason::EndTurn)
            .usage(
                sdk::TokenUsage::builder()
                    .input_tokens(15)
                    .output_tokens(5)
                    .total_tokens(20)
                    .cache_read_input_tokens(3)
                    .build()
                    .unwrap(),
            )
            .metrics(
                sdk::ConverseMetrics::builder()
                    .latency_ms(700)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        let response = from_sdk_output(&output).unwrap();

        assert_eq!(
            response.message,
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text("Ready.".into())],
            }
        );
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 15,
                output_tokens: 5,
                cache_read_tokens: 3,
                cache_write_tokens: 0,
            }
        );
        assert_eq!(response.latency, Duration::from_millis(700));
    }

    #[test]
    fn bedrock_exceptions_map_to_what_the_caller_can_do() {
        use aws_sdk_bedrockruntime::types::error::{
            AccessDeniedException, InternalServerException, ModelNotReadyException,
            ResourceNotFoundException, ThrottlingException, ValidationException,
        };

        let throttled = ConverseError::ThrottlingException(
            ThrottlingException::builder()
                .message("Too many requests")
                .build(),
        );
        let not_ready = ConverseError::ModelNotReadyException(
            ModelNotReadyException::builder().message("Loading").build(),
        );
        let internal = ConverseError::InternalServerException(
            InternalServerException::builder().message("Oops").build(),
        );
        let denied = ConverseError::AccessDeniedException(
            AccessDeniedException::builder()
                .message("Not available for this account")
                .build(),
        );
        let invalid = ConverseError::ValidationException(
            ValidationException::builder()
                .message("The provided model identifier is invalid.")
                .build(),
        );
        let not_found = ConverseError::ResourceNotFoundException(
            ResourceNotFoundException::builder()
                .message("Model not found")
                .build(),
        );

        assert!(matches!(
            from_converse_error(throttled),
            LlmError::Throttled(message) if message.contains("Too many requests")
        ));
        assert!(matches!(
            from_converse_error(not_ready),
            LlmError::ModelNotReady(_)
        ));
        assert!(matches!(
            from_converse_error(internal),
            LlmError::ServiceUnavailable(_)
        ));
        assert!(matches!(
            from_converse_error(denied),
            LlmError::AccessDenied(_)
        ));
        assert!(matches!(
            from_converse_error(invalid),
            LlmError::ValidationError(_)
        ));
        assert!(matches!(
            from_converse_error(not_found),
            LlmError::ValidationError(_)
        ));
    }

    #[test]
    fn a_validation_error_about_the_input_length_means_context_too_long() {
        use aws_sdk_bedrockruntime::types::error::ValidationException;

        let too_long = ConverseError::ValidationException(
            ValidationException::builder()
                .message(
                    "The model returned the following errors: \
                     prompt is too long: 260024 tokens > 200000 maximum",
                )
                .build(),
        );

        assert!(matches!(
            from_converse_error(too_long),
            LlmError::ContextTooLong(_)
        ));
    }

    #[test]
    fn a_stop_reason_without_a_neutral_equivalent_keeps_its_name() {
        assert_eq!(
            from_sdk_stop_reason(&sdk::StopReason::ToolUse),
            StopReason::Other("tool_use".into())
        );
    }
}
