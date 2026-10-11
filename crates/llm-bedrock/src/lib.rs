//! The Amazon Bedrock implementation of [`llm_core::LlmClient`], over the Converse API.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::time::Duration;

use aws_config::retry::RetryConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::{DisplayErrorContext, SdkError};
use aws_sdk_bedrockruntime::operation::converse::{ConverseError, ConverseOutput};
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::primitives::event_stream::EventReceiver;
use aws_sdk_bedrockruntime::types as sdk;
use aws_sdk_bedrockruntime::types::error::{ConverseStreamOutputError, ValidationException};
use aws_smithy_types::{Document, Number};
use futures::StreamExt;
use futures::stream::BoxStream;
use llm_core::{
    AssistantBlock, LlmClient, LlmError, LlmEvent, LlmRequest, LlmResponse, Message, ModelId,
    OutputSchema, StopReason, SystemBlock, ToolOutcome, ToolSpec, Usage, UserBlock,
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
    fn prepare(&self, mut req: LlmRequest) -> LlmRequest {
        for path in rejected_params(&req.model) {
            if remove_path(&mut req.extra, path) {
                tracing::warn!(
                    model = req.model.as_str(),
                    parameter = path.join("."),
                    "the model does not accept this parameter, so it was not sent"
                );
            }
        }
        req
    }

    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        // Prepared here too, so that the client is safe without any decorator around it.
        self.converse(&self.prepare(req)).await
    }

    async fn stream(
        &self,
        req: LlmRequest,
    ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
        let receiver = self.start_stream(&self.prepare(req)).await?;

        // The SDK hands out a receiver to pull events from, not a `Stream`: `unfold` turns one
        // into the other. The state becomes `None` after an error, which ends the stream.
        let events = futures::stream::unfold(Some(receiver), |state| async move {
            let mut receiver = state?;
            match next_event(&mut receiver).await? {
                Ok(event) => Some((Ok(event), Some(receiver))),
                Err(err) => Some((Err(err), None)),
            }
        });
        Ok(events.boxed())
    }
}

/// What a model family rejects, as paths into `LlmRequest.extra`, matched on a part of the model
/// id. A model missing from the table gets every parameter as it is. In Module 10 this table
/// moves into the `ModelRegistry`.
const REJECTED_PARAMS: &[(&str, &[&[&str]])] = &[
    // Bedrock answers "This model does not support the effort parameter".
    ("claude-haiku-4-5", &[&["output_config", "effort"]]),
];

fn rejected_params(model: &ModelId) -> impl Iterator<Item = &'static [&'static str]> {
    REJECTED_PARAMS
        .iter()
        .filter(|(family, _)| model.as_str().contains(family))
        .flat_map(|(_, paths)| paths.iter().copied())
}

/// Removes the field at `path`, and every object that removing it leaves empty, so that no empty
/// object reaches the model. Says whether the field was there.
fn remove_path(value: &mut serde_json::Value, path: &[&str]) -> bool {
    let Some(fields) = value.as_object_mut() else {
        return false;
    };
    match path {
        [] => false,
        [last] => fields.remove(*last).is_some(),
        [first, rest @ ..] => {
            let Some(child) = fields.get_mut(*first) else {
                return false;
            };
            let removed = remove_path(child, rest);
            if removed && child.as_object().is_some_and(|child| child.is_empty()) {
                fields.remove(*first);
            }
            removed
        }
    }
}

type StreamReceiver = EventReceiver<sdk::ConverseStreamOutput, ConverseStreamOutputError>;

impl BedrockClient {
    async fn converse(&self, req: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let request = SdkRequest::try_from(req)?;
        let output = self
            .client
            .converse()
            .model_id(req.model.as_str())
            .set_system(request.system)
            .set_messages(Some(request.messages))
            .inference_config(request.inference_config)
            .set_additional_model_request_fields(request.additional_fields)
            .set_output_config(request.output_config)
            .set_tool_config(request.tool_config)
            .send()
            .await
            .map_err(|err| from_sdk_error(err, from_converse_error))?;
        from_sdk_output(&output)
    }

    async fn start_stream(&self, req: &LlmRequest) -> Result<StreamReceiver, LlmError> {
        let request = SdkRequest::try_from(req)?;
        let output = self
            .client
            .converse_stream()
            .model_id(req.model.as_str())
            .set_system(request.system)
            .set_messages(Some(request.messages))
            .inference_config(request.inference_config)
            .set_additional_model_request_fields(request.additional_fields)
            .set_output_config(request.output_config)
            .set_tool_config(request.tool_config)
            .send()
            .await
            .map_err(|err| from_sdk_error(err, from_converse_stream_error))?;
        Ok(output.stream)
    }
}

/// The next event worth passing on, skipping the ones `from_stream_event` ignores; `None` when
/// the stream is over.
async fn next_event(receiver: &mut StreamReceiver) -> Option<Result<LlmEvent, LlmError>> {
    loop {
        match receiver.recv().await {
            Ok(Some(event)) => match from_stream_event(&event) {
                Ok(Some(event)) => return Some(Ok(event)),
                Ok(None) => continue,
                Err(err) => return Some(Err(err)),
            },
            Ok(None) => return None,
            Err(err) => return Some(Err(from_sdk_error(err, from_stream_output_error))),
        }
    }
}

/// The parts of a Converse request that `converse` and `converse_stream` share.
struct SdkRequest {
    system: Option<Vec<sdk::SystemContentBlock>>,
    messages: Vec<sdk::Message>,
    inference_config: sdk::InferenceConfiguration,
    /// Model-specific fields, such as effort for Claude, which Bedrock passes to the model as they
    /// are. The fields every model shares go in `inference_config` instead.
    additional_fields: Option<Document>,
    /// The JSON Schema the reply must follow, when the caller asked for structured output.
    output_config: Option<sdk::OutputConfig>,
    /// The tools the model may ask for.
    tool_config: Option<sdk::ToolConfiguration>,
}

impl TryFrom<&LlmRequest> for SdkRequest {
    type Error = LlmError;

    fn try_from(req: &LlmRequest) -> Result<Self, LlmError> {
        let system = to_sdk_system(&req.system)?;
        let messages = req
            .messages
            .iter()
            .map(to_sdk_message)
            .collect::<Result<Vec<_>, _>>()?;
        let max_tokens = i32::try_from(req.max_tokens).map_err(invalid)?;
        let additional_fields = match &req.extra {
            serde_json::Value::Null => None,
            extra @ serde_json::Value::Object(_) => Some(to_document(extra)),
            other => {
                return Err(LlmError::ValidationError(format!(
                    "extra must be a JSON object or null, not {other}"
                )));
            }
        };
        let output_config = req.output_schema.as_ref().map(to_sdk_output).transpose()?;
        Ok(Self {
            system: (!system.is_empty()).then_some(system),
            messages,
            inference_config: sdk::InferenceConfiguration::builder()
                .max_tokens(max_tokens)
                .build(),
            additional_fields,
            output_config,
            // Converse rejects a tool configuration with no tools in it.
            tool_config: (!req.tools.is_empty())
                .then(|| to_sdk_tools(&req.tools))
                .transpose()?,
        })
    }
}

fn to_sdk_tools(tools: &[ToolSpec]) -> Result<sdk::ToolConfiguration, LlmError> {
    let tools = tools
        .iter()
        .map(|tool| {
            sdk::ToolSpecification::builder()
                .name(tool.name.as_str())
                .description(tool.description.as_str())
                .input_schema(sdk::ToolInputSchema::Json(object_document(
                    tool.input_schema.as_object(),
                )))
                .build()
                .map(sdk::Tool::ToolSpec)
                .map_err(invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    sdk::ToolConfiguration::builder()
        .set_tools(Some(tools))
        .build()
        .map_err(invalid)
}

/// Structured output through Converse's own field, `outputConfig.textFormat`, which takes the
/// schema as a JSON string.
fn to_sdk_output(output: &OutputSchema) -> Result<sdk::OutputConfig, LlmError> {
    let definition = sdk::JsonSchemaDefinition::builder()
        .name(&output.name)
        .schema(output.schema.to_string())
        .build()
        .map_err(invalid)?;
    let format = sdk::OutputFormat::builder()
        .r#type(sdk::OutputFormatType::JsonSchema)
        .structure(sdk::OutputFormatStructure::JsonSchema(definition))
        .build()
        .map_err(invalid)?;
    Ok(sdk::OutputConfig::builder().text_format(format).build())
}

/// Converts JSON into the SDK's own JSON type, which Converse expects for model-specific fields.
fn to_document(value: &serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(value) => Document::Bool(*value),
        serde_json::Value::Number(number) => match (number.as_u64(), number.as_i64()) {
            (Some(value), _) => Document::Number(Number::PosInt(value)),
            (None, Some(value)) => Document::Number(Number::NegInt(value)),
            (None, None) => number.as_f64().map_or(Document::Null, |value| {
                Document::Number(Number::Float(value))
            }),
        },
        serde_json::Value::String(value) => Document::String(value.clone()),
        serde_json::Value::Array(values) => {
            Document::Array(values.iter().map(to_document).collect())
        }
        serde_json::Value::Object(fields) => object_document(fields),
    }
}

fn object_document(fields: &serde_json::Map<String, serde_json::Value>) -> Document {
    Document::Object(
        fields
            .iter()
            .map(|(key, value)| (key.clone(), to_document(value)))
            .collect::<HashMap<_, _>>(),
    )
}

/// A request the SDK refuses to build is a bug in the request, not something to retry.
fn invalid(err: impl Display) -> LlmError {
    LlmError::ValidationError(err.to_string())
}

/// Maps an error that is left after the SDK's own retries. Exceptions returned by Bedrock go to
/// `from_service`; the rest are failures to reach it.
fn from_sdk_error<E, R>(err: SdkError<E, R>, from_service: impl FnOnce(E) -> LlmError) -> LlmError
where
    E: Error + 'static,
    R: Debug,
{
    match err {
        SdkError::ServiceError(context) => from_service(context.into_err()),
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

// Bedrock has three error enums with nearly the same exceptions: one for `converse`, one for
// starting `converse_stream`, and one for failures in the middle of a stream. Each gets its own
// `match`, so that the compiler checks the variants of each. The `Display` of an exception is its
// name followed by its message, which is short enough for logs.

fn from_converse_error(err: ConverseError) -> LlmError {
    let message = err.to_string();
    match err {
        ConverseError::ThrottlingException(_) => LlmError::Throttled(message),
        ConverseError::ModelNotReadyException(_) => LlmError::ModelNotReady(message),
        ConverseError::ServiceUnavailableException(_)
        | ConverseError::InternalServerException(_)
        | ConverseError::ModelTimeoutException(_) => LlmError::ServiceUnavailable(message),
        ConverseError::AccessDeniedException(_) => LlmError::AccessDenied(message),
        ConverseError::ValidationException(exception) => from_validation(&exception, message),
        ConverseError::ResourceNotFoundException(_) => LlmError::ValidationError(message),
        _ => LlmError::Other(message),
    }
}

fn from_converse_stream_error(err: ConverseStreamError) -> LlmError {
    let message = err.to_string();
    match err {
        ConverseStreamError::ThrottlingException(_) => LlmError::Throttled(message),
        ConverseStreamError::ModelNotReadyException(_) => LlmError::ModelNotReady(message),
        ConverseStreamError::ServiceUnavailableException(_)
        | ConverseStreamError::InternalServerException(_)
        | ConverseStreamError::ModelTimeoutException(_)
        | ConverseStreamError::ModelStreamErrorException(_) => {
            LlmError::ServiceUnavailable(message)
        }
        ConverseStreamError::AccessDeniedException(_) => LlmError::AccessDenied(message),
        ConverseStreamError::ValidationException(exception) => from_validation(&exception, message),
        ConverseStreamError::ResourceNotFoundException(_) => LlmError::ValidationError(message),
        _ => LlmError::Other(message),
    }
}

/// A failure after the stream has started. The SDK does not retry it: part of the reply has
/// already arrived, so the caller decides.
fn from_stream_output_error(err: ConverseStreamOutputError) -> LlmError {
    let message = err.to_string();
    match err {
        ConverseStreamOutputError::ThrottlingException(_) => LlmError::Throttled(message),
        ConverseStreamOutputError::ServiceUnavailableException(_)
        | ConverseStreamOutputError::InternalServerException(_)
        | ConverseStreamOutputError::ModelStreamErrorException(_) => {
            LlmError::ServiceUnavailable(message)
        }
        ConverseStreamOutputError::ValidationException(exception) => {
            from_validation(&exception, message)
        }
        _ => LlmError::Other(message),
    }
}

/// Bedrock reports an input that does not fit the context window as a plain validation error:
/// only its message tells the two apart.
fn from_validation(exception: &ValidationException, message: String) -> LlmError {
    let too_long = exception
        .message()
        .is_some_and(|text| text.to_lowercase().contains("too long"));
    if too_long {
        LlmError::ContextTooLong(message)
    } else {
        LlmError::ValidationError(message)
    }
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
    let (role, content) = match message {
        Message::User(blocks) => (
            sdk::ConversationRole::User,
            blocks
                .iter()
                .map(to_sdk_user_block)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Message::Assistant(blocks) => (
            sdk::ConversationRole::Assistant,
            blocks
                .iter()
                .map(to_sdk_assistant_block)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    sdk::Message::builder()
        .role(role)
        .set_content(Some(content))
        .build()
        .map_err(invalid)
}

fn to_sdk_user_block(block: &UserBlock) -> Result<sdk::ContentBlock, LlmError> {
    match block {
        UserBlock::Text(text) => Ok(sdk::ContentBlock::Text(text.clone())),
        UserBlock::CachePoint => cache_point().map(sdk::ContentBlock::CachePoint),
        UserBlock::ToolResult {
            tool_use_id,
            outcome,
        } => {
            let (status, text) = match outcome {
                ToolOutcome::Success(text) => (sdk::ToolResultStatus::Success, text),
                ToolOutcome::Failure(text) => (sdk::ToolResultStatus::Error, text),
            };
            sdk::ToolResultBlock::builder()
                .tool_use_id(tool_use_id.as_str())
                .content(sdk::ToolResultContentBlock::Text(text.clone()))
                .status(status)
                .build()
                .map(sdk::ContentBlock::ToolResult)
                .map_err(invalid)
        }
    }
}

fn to_sdk_assistant_block(block: &AssistantBlock) -> Result<sdk::ContentBlock, LlmError> {
    match block {
        AssistantBlock::Text(text) => Ok(sdk::ContentBlock::Text(text.clone())),
        AssistantBlock::ToolUse { id, name, input } => sdk::ToolUseBlock::builder()
            .tool_use_id(id.as_str())
            .name(name.as_str())
            .input(object_document(input))
            .build()
            .map(sdk::ContentBlock::ToolUse)
            .map_err(invalid),
    }
}

fn from_sdk_output(output: &ConverseOutput) -> Result<LlmResponse, LlmError> {
    let Some(sdk::ConverseOutput::Message(message)) = output.output() else {
        return Err(LlmError::Other("the response contains no message".into()));
    };
    let metrics = output
        .metrics()
        .ok_or_else(|| LlmError::Other("the response contains no metrics".into()))?;

    Ok(LlmResponse {
        content: from_sdk_content(message),
        stop_reason: from_sdk_stop_reason(output.stop_reason()),
        usage: from_sdk_usage(output.usage())?,
        latency: latency(metrics.latency_ms())?,
    })
}

/// Translates one stream event. Only text, the stop reason and the final counts matter for now;
/// the other events, such as the start and end of a block, give `None`.
fn from_stream_event(event: &sdk::ConverseStreamOutput) -> Result<Option<LlmEvent>, LlmError> {
    match event {
        sdk::ConverseStreamOutput::ContentBlockDelta(delta) => match delta.delta() {
            Some(sdk::ContentBlockDelta::Text(text)) => Ok(Some(LlmEvent::TextDelta(text.clone()))),
            _ => Ok(None),
        },
        sdk::ConverseStreamOutput::MessageStop(stop) => Ok(Some(LlmEvent::Stop(
            from_sdk_stop_reason(stop.stop_reason()),
        ))),
        sdk::ConverseStreamOutput::Metadata(metadata) => {
            let metrics = metadata
                .metrics()
                .ok_or_else(|| LlmError::Other("the stream metadata contains no metrics".into()))?;
            Ok(Some(LlmEvent::Metadata {
                usage: from_sdk_usage(metadata.usage())?,
                latency: latency(metrics.latency_ms())?,
            }))
        }
        _ => Ok(None),
    }
}

fn from_sdk_usage(usage: Option<&sdk::TokenUsage>) -> Result<Usage, LlmError> {
    let usage =
        usage.ok_or_else(|| LlmError::Other("the response contains no token usage".into()))?;
    Ok(Usage {
        input_tokens: token_count(usage.input_tokens())?,
        output_tokens: token_count(usage.output_tokens())?,
        cache_read_tokens: token_count(usage.cache_read_input_tokens().unwrap_or(0))?,
        cache_write_tokens: token_count(usage.cache_write_input_tokens().unwrap_or(0))?,
    })
}

fn latency(milliseconds: i64) -> Result<Duration, LlmError> {
    u64::try_from(milliseconds)
        .map(Duration::from_millis)
        .map_err(|_| LlmError::Other(format!("negative latency {milliseconds}")))
}

/// The blocks of the model's reply. Its role is not read: Converse replies as the assistant.
fn from_sdk_content(message: &sdk::Message) -> Vec<AssistantBlock> {
    // Only text is kept for now: reasoning blocks, which Claude 5 models emit by default, and
    // every other kind of block are skipped.
    message
        .content()
        .iter()
        .filter_map(|block| match block {
            sdk::ContentBlock::Text(text) => Some(AssistantBlock::Text(text.clone())),
            _ => None,
        })
        .collect()
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
    use llm_core::{ToolName, ToolUseId};

    use super::*;

    #[test]
    fn a_message_keeps_its_role_and_blocks_in_the_sdk() {
        let message = Message::User(vec![
            UserBlock::Text("ready?".into()),
            UserBlock::CachePoint,
        ]);

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
    fn a_tool_request_keeps_its_id_name_and_input_in_the_sdk() {
        let input = serde_json::json!({ "time_zone": "Europe/Rome" });
        let message = Message::Assistant(vec![AssistantBlock::ToolUse {
            id: ToolUseId::new("tooluse_kZJMlvQmRJ6eAyJE5GIl7Q"),
            name: ToolName::new("get_current_date").unwrap(),
            input: input.as_object().unwrap().clone(),
        }]);

        let sdk_message = to_sdk_message(&message).unwrap();

        assert_eq!(sdk_message.role(), &sdk::ConversationRole::Assistant);
        let [sdk::ContentBlock::ToolUse(tool_use)] = sdk_message.content() else {
            panic!("expected one tool use, got {:?}", sdk_message.content());
        };
        assert_eq!(tool_use.tool_use_id(), "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q");
        assert_eq!(tool_use.name(), "get_current_date");
        assert_eq!(
            tool_use.input(),
            &Document::Object(HashMap::from([(
                "time_zone".to_owned(),
                Document::String("Europe/Rome".into()),
            )]))
        );
    }

    fn sdk_tool_result(outcome: ToolOutcome) -> sdk::ToolResultBlock {
        let message = Message::User(vec![UserBlock::ToolResult {
            tool_use_id: ToolUseId::new("tooluse_kZJMlvQmRJ6eAyJE5GIl7Q"),
            outcome,
        }]);
        let sdk_message = to_sdk_message(&message).unwrap();
        let [sdk::ContentBlock::ToolResult(result)] = sdk_message.content() else {
            panic!("expected one tool result, got {:?}", sdk_message.content());
        };
        result.clone()
    }

    #[test]
    fn a_successful_tool_result_is_sent_with_the_success_status() {
        let result = sdk_tool_result(ToolOutcome::Success("2026-10-12".into()));

        assert_eq!(result.tool_use_id(), "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q");
        assert_eq!(
            result.content(),
            [sdk::ToolResultContentBlock::Text("2026-10-12".into())]
        );
        assert_eq!(result.status(), Some(&sdk::ToolResultStatus::Success));
    }

    #[test]
    fn a_failed_tool_result_is_sent_with_the_error_status() {
        let result = sdk_tool_result(ToolOutcome::Failure(
            "invalid input: missing field `time_zone`".into(),
        ));

        assert_eq!(result.tool_use_id(), "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q");
        assert_eq!(
            result.content(),
            [sdk::ToolResultContentBlock::Text(
                "invalid input: missing field `time_zone`".into()
            )]
        );
        assert_eq!(result.status(), Some(&sdk::ToolResultStatus::Error));
    }

    #[test]
    fn a_cache_point_closes_the_system_prompt_prefix() {
        let system = to_sdk_system(&[
            SystemBlock::Text("You are the assistant of a fictional clinic.".into()),
            SystemBlock::CachePoint,
        ])
        .unwrap();

        assert!(matches!(&system[0], sdk::SystemContentBlock::Text(_)));
        assert!(matches!(
            &system[1],
            sdk::SystemContentBlock::CachePoint(point) if point.r#type() == &sdk::CachePointType::Default
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

        assert_eq!(response.content, [AssistantBlock::Text("Ready.".into())]);
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
            ResourceNotFoundException, ThrottlingException,
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
    fn a_failure_in_the_middle_of_a_stream_maps_like_the_others() {
        use aws_sdk_bedrockruntime::types::error::{
            ModelStreamErrorException, ThrottlingException,
        };

        let throttled = ConverseStreamOutputError::ThrottlingException(
            ThrottlingException::builder().message("Slow down").build(),
        );
        let broken = ConverseStreamOutputError::ModelStreamErrorException(
            ModelStreamErrorException::builder()
                .message("An error occurred while streaming")
                .build(),
        );

        assert!(matches!(
            from_stream_output_error(throttled),
            LlmError::Throttled(_)
        ));
        assert!(matches!(
            from_stream_output_error(broken),
            LlmError::ServiceUnavailable(_)
        ));
    }

    #[test]
    fn stream_events_become_text_deltas_a_stop_and_the_final_counts() {
        let text = sdk::ConverseStreamOutput::ContentBlockDelta(
            sdk::ContentBlockDeltaEvent::builder()
                .content_block_index(0)
                .delta(sdk::ContentBlockDelta::Text("Rea".into()))
                .build()
                .unwrap(),
        );
        let reasoning = sdk::ConverseStreamOutput::ContentBlockDelta(
            sdk::ContentBlockDeltaEvent::builder()
                .content_block_index(0)
                .delta(sdk::ContentBlockDelta::ReasoningContent(
                    sdk::ReasoningContentBlockDelta::Text("Thinking".into()),
                ))
                .build()
                .unwrap(),
        );
        let start = sdk::ConverseStreamOutput::MessageStart(
            sdk::MessageStartEvent::builder()
                .role(sdk::ConversationRole::Assistant)
                .build()
                .unwrap(),
        );
        let stop = sdk::ConverseStreamOutput::MessageStop(
            sdk::MessageStopEvent::builder()
                .stop_reason(sdk::StopReason::MaxTokens)
                .build()
                .unwrap(),
        );
        let metadata = sdk::ConverseStreamOutput::Metadata(
            sdk::ConverseStreamMetadataEvent::builder()
                .usage(
                    sdk::TokenUsage::builder()
                        .input_tokens(15)
                        .output_tokens(5)
                        .total_tokens(20)
                        .build()
                        .unwrap(),
                )
                .metrics(
                    sdk::ConverseStreamMetrics::builder()
                        .latency_ms(650)
                        .build()
                        .unwrap(),
                )
                .build(),
        );

        assert_eq!(
            from_stream_event(&text).unwrap(),
            Some(LlmEvent::TextDelta("Rea".into()))
        );
        assert_eq!(from_stream_event(&reasoning).unwrap(), None);
        assert_eq!(from_stream_event(&start).unwrap(), None);
        assert_eq!(
            from_stream_event(&stop).unwrap(),
            Some(LlmEvent::Stop(StopReason::MaxTokens))
        );
        assert_eq!(
            from_stream_event(&metadata).unwrap(),
            Some(LlmEvent::Metadata {
                usage: Usage {
                    input_tokens: 15,
                    output_tokens: 5,
                    ..Usage::default()
                },
                latency: Duration::from_millis(650),
            })
        );
    }

    fn request_with_extra(extra: serde_json::Value) -> LlmRequest {
        LlmRequest {
            model: llm_core::ModelId::new("test-model"),
            system: vec![],
            messages: vec![Message::User(vec![UserBlock::Text("ready?".into())])],
            max_tokens: 50,
            extra,
            prompt: None,
            tools: vec![],
            output_schema: None,
        }
    }

    #[test]
    fn extra_json_becomes_an_sdk_document_with_the_same_shape() {
        let extra = serde_json::json!({
            "output_config": { "effort": "low" },
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "stop": ["END", null],
            "temperature": 0.5,
            "offset": -3,
            "strict": true,
        });

        let request = SdkRequest::try_from(&request_with_extra(extra)).unwrap();

        let expected = Document::Object(HashMap::from([
            (
                "output_config".to_owned(),
                Document::Object(HashMap::from([(
                    "effort".to_owned(),
                    Document::String("low".into()),
                )])),
            ),
            (
                "thinking".to_owned(),
                Document::Object(HashMap::from([
                    ("type".to_owned(), Document::String("enabled".into())),
                    (
                        "budget_tokens".to_owned(),
                        Document::Number(Number::PosInt(1024)),
                    ),
                ])),
            ),
            (
                "stop".to_owned(),
                Document::Array(vec![Document::String("END".into()), Document::Null]),
            ),
            (
                "temperature".to_owned(),
                Document::Number(Number::Float(0.5)),
            ),
            ("offset".to_owned(), Document::Number(Number::NegInt(-3))),
            ("strict".to_owned(), Document::Bool(true)),
        ]));
        assert_eq!(request.additional_fields, Some(expected));
    }

    #[test]
    fn null_extra_sends_no_additional_fields() {
        let request = SdkRequest::try_from(&request_with_extra(serde_json::Value::Null)).unwrap();

        assert_eq!(request.additional_fields, None);
    }

    #[test]
    fn extra_that_is_not_an_object_is_an_invalid_request() {
        let result = SdkRequest::try_from(&request_with_extra(serde_json::json!("low")));

        assert!(matches!(result, Err(LlmError::ValidationError(_))));
    }

    /// A client that is never sent anything: building it reads no AWS configuration.
    fn offline_client() -> BedrockClient {
        let config = aws_sdk_bedrockruntime::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new("eu-west-1"))
            .build();
        BedrockClient {
            client: aws_sdk_bedrockruntime::Client::from_conf(config),
        }
    }

    fn request_for(model: &str, extra: serde_json::Value) -> LlmRequest {
        LlmRequest {
            model: ModelId::new(model),
            ..request_with_extra(extra)
        }
    }

    const HAIKU_4_5: &str = "eu.anthropic.claude-haiku-4-5-20251001-v1:0";

    #[test]
    fn effort_is_dropped_for_a_model_that_rejects_it() {
        let request = request_for(
            HAIKU_4_5,
            serde_json::json!({
                "output_config": { "effort": "low", "format": "json" },
                "thinking": { "type": "enabled", "budget_tokens": 1024 },
            }),
        );

        let prepared = offline_client().prepare(request);

        assert_eq!(
            prepared.extra,
            serde_json::json!({
                "output_config": { "format": "json" },
                "thinking": { "type": "enabled", "budget_tokens": 1024 },
            })
        );
    }

    #[test]
    fn an_object_left_empty_by_a_dropped_parameter_is_not_sent() {
        let request = request_for(
            HAIKU_4_5,
            serde_json::json!({ "output_config": { "effort": "low" } }),
        );

        let prepared = offline_client().prepare(request);

        assert_eq!(prepared.extra, serde_json::json!({}));
    }

    #[test]
    fn a_model_that_accepts_effort_gets_the_request_unchanged() {
        let request = request_for(
            "eu.anthropic.claude-sonnet-4-6",
            serde_json::json!({ "output_config": { "effort": "low" } }),
        );

        let prepared = offline_client().prepare(request.clone());

        assert_eq!(prepared, request);
    }

    #[test]
    fn an_output_schema_becomes_a_json_schema_text_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "label": { "type": "string", "enum": ["SAFE", "PSYCH_CRISIS"] } },
            "required": ["label"],
            "additionalProperties": false,
        });
        let request = LlmRequest {
            output_schema: Some(llm_core::OutputSchema {
                name: "Verdict".into(),
                schema: schema.clone(),
            }),
            ..request_with_extra(serde_json::Value::Null)
        };

        let request = SdkRequest::try_from(&request).unwrap();

        let format = request
            .output_config
            .and_then(|config| config.text_format)
            .expect("no text format");
        assert_eq!(format.r#type(), &sdk::OutputFormatType::JsonSchema);
        let definition = format.structure().unwrap().as_json_schema().unwrap();
        assert_eq!(definition.name(), Some("Verdict"));
        let sent: serde_json::Value = serde_json::from_str(definition.schema()).unwrap();
        assert_eq!(sent, schema);
    }

    #[test]
    fn without_an_output_schema_no_output_config_is_sent() {
        let request = SdkRequest::try_from(&request_with_extra(serde_json::Value::Null)).unwrap();

        assert!(request.output_config.is_none());
    }

    #[test]
    fn without_tools_no_tool_configuration_is_sent() {
        let request = SdkRequest::try_from(&request_with_extra(serde_json::Value::Null)).unwrap();

        // Converse rejects a tool configuration with no tools in it.
        assert!(request.tool_config.is_none());
    }

    #[test]
    fn a_tool_becomes_a_tool_specification_with_its_schema_as_a_document() {
        let request = LlmRequest {
            tools: vec![llm_core::ToolSpec {
                name: llm_core::ToolName::new("get_current_date").unwrap(),
                description: llm_core::ToolDescription::new(
                    "Today's date in the clinic's time zone.",
                )
                .unwrap(),
                input_schema: llm_core::InputSchema::new(serde_json::json!({
                    "type": "object",
                    "properties": { "time_zone": { "type": "string" } },
                }))
                .unwrap(),
            }],
            ..request_with_extra(serde_json::Value::Null)
        };

        let request = SdkRequest::try_from(&request).unwrap();

        let tools = request.tool_config.expect("a tool configuration").tools;
        let [sdk::Tool::ToolSpec(spec)] = tools.as_slice() else {
            panic!("expected one tool specification, got {tools:?}");
        };
        assert_eq!(spec.name(), "get_current_date");
        assert_eq!(
            spec.description(),
            Some("Today's date in the clinic's time zone.")
        );
        let expected = Document::Object(HashMap::from([
            ("type".to_owned(), Document::String("object".into())),
            (
                "properties".to_owned(),
                Document::Object(HashMap::from([(
                    "time_zone".to_owned(),
                    Document::Object(HashMap::from([(
                        "type".to_owned(),
                        Document::String("string".into()),
                    )])),
                )])),
            ),
        ]));
        assert_eq!(
            spec.input_schema(),
            Some(&sdk::ToolInputSchema::Json(expected))
        );
    }

    #[test]
    fn a_stop_reason_without_a_neutral_equivalent_keeps_its_name() {
        assert_eq!(
            from_sdk_stop_reason(&sdk::StopReason::ToolUse),
            StopReason::Other("tool_use".into())
        );
    }
}
