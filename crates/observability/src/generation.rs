//! What turns the span of a model call into a Langfuse generation. The caller records its
//! neutral types; the `langfuse.*` attribute names live only here.

use std::time::SystemTime;

use llm_core::{
    ContentBlock, LlmError, LlmRequest, LlmResponse, Message, Role, SystemBlock, Usage,
};
use serde_json::{Value, json};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Marks `span` as a generation and records the request and the response.
pub(crate) fn record_generation(span: &Span, req: &LlmRequest, resp: &LlmResponse) {
    record_request(span, req);
    span.set_attribute(
        "langfuse.observation.output",
        message(&resp.message).to_string(),
    );
    span.set_attribute(
        "langfuse.observation.usage_details",
        usage_details(&resp.usage).to_string(),
    );
    span.set_attribute(
        "langfuse.observation.metadata.stop_reason",
        format!("{:?}", resp.stop_reason),
    );
    // Metadata values are strings in the Langfuse attribute mapping.
    span.set_attribute(
        "langfuse.observation.metadata.provider_latency_ms",
        resp.latency.as_millis().to_string(),
    );
}

/// Marks `span` as a failed generation: the request, the error level and the error text.
pub(crate) fn record_generation_error(span: &Span, req: &LlmRequest, err: &LlmError) {
    record_request(span, req);
    span.set_attribute("langfuse.observation.level", "ERROR");
    span.set_attribute("langfuse.observation.status_message", err.to_string());
}

/// Records when the first token of a stream arrived, which Langfuse shows as time to first token.
pub(crate) fn record_completion_start(span: &Span, at: SystemTime) {
    span.set_attribute(
        "langfuse.observation.completion_start_time",
        humantime::format_rfc3339_millis(at).to_string(),
    );
}

fn record_request(span: &Span, req: &LlmRequest) {
    span.set_attribute("langfuse.observation.type", "generation");
    // The id must match a model definition in Langfuse (step 5.4), or the cost stays empty.
    span.set_attribute(
        "langfuse.observation.model.name",
        req.model.as_str().to_owned(),
    );
    span.set_attribute(
        "langfuse.observation.model.parameters",
        model_parameters(req).to_string(),
    );
    span.set_attribute("langfuse.observation.input", input(req).to_string());
    // Links the generation to that prompt version in Langfuse, which then shows cost and latency
    // per version. The version must be an integer, or Langfuse does not link it.
    if let Some(prompt) = &req.prompt {
        span.set_attribute("langfuse.observation.prompt.name", prompt.name.clone());
        span.set_attribute(
            "langfuse.observation.prompt.version",
            i64::from(prompt.version),
        );
    }
}

/// The request as a list of `role`/`content` messages, the shape Langfuse renders as a
/// conversation. The system prompt comes first, as a message of its own.
fn input(req: &LlmRequest) -> Value {
    let system: String = req
        .system
        .iter()
        .filter_map(|block| match block {
            SystemBlock::Text(text) => Some(text.as_str()),
            SystemBlock::CachePoint => None,
        })
        .collect();
    let system = (!system.is_empty()).then(|| json!({ "role": "system", "content": system }));
    Value::Array(
        system
            .into_iter()
            .chain(req.messages.iter().map(message))
            .collect(),
    )
}

fn message(message: &Message) -> Value {
    let role = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };
    let content: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            ContentBlock::CachePoint => None,
        })
        .collect();
    json!({ "role": role, "content": content })
}

/// `max_tokens`, plus the model-specific fields such as effort.
fn model_parameters(req: &LlmRequest) -> Value {
    let mut parameters = serde_json::Map::new();
    parameters.insert("max_tokens".to_owned(), req.max_tokens.into());
    if let Value::Object(extra) = &req.extra {
        parameters.extend(extra.clone());
    }
    Value::Object(parameters)
}

/// The keys are the ones the model definitions of step 5.4 put a price on: a key without a price
/// is counted but costs nothing.
fn usage_details(usage: &Usage) -> Value {
    json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "cache_creation_input_tokens": usage.cache_write_tokens,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use llm_core::{ModelId, StopReason};

    use super::*;
    use crate::test_support::exported_spans;

    fn request() -> LlmRequest {
        LlmRequest {
            model: ModelId::new("eu.anthropic.claude-haiku-4-5-20251001-v1:0"),
            system: vec![
                SystemBlock::Text("You are terse.".into()),
                SystemBlock::CachePoint,
            ],
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text("Ready?".into())],
            }],
            max_tokens: 50,
            extra: json!({ "effort": "low" }),
            prompt: None,
            output_schema: None,
        }
    }

    fn response() -> LlmResponse {
        LlmResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text("Ready.".into())],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 15,
                output_tokens: 5,
                cache_read_tokens: 2,
                cache_write_tokens: 3,
            },
            latency: Duration::from_millis(700),
        }
    }

    /// Runs `record` inside one span and returns the attributes that span was exported with.
    fn exported_attributes(record: impl FnOnce(&Span)) -> HashMap<String, String> {
        let mut spans = exported_spans(|| record(&tracing::info_span!("call-model")));
        assert_eq!(spans.len(), 1);
        spans.remove(0).attributes
    }

    #[test]
    fn a_completed_call_becomes_a_generation_with_model_usage_and_messages() {
        let attributes =
            exported_attributes(|span| record_generation(span, &request(), &response()));

        assert_eq!(attributes["langfuse.observation.type"], "generation");
        assert_eq!(
            attributes["langfuse.observation.model.name"],
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        );
        let parameters: Value =
            serde_json::from_str(&attributes["langfuse.observation.model.parameters"]).unwrap();
        assert_eq!(parameters, json!({ "max_tokens": 50, "effort": "low" }));
        let input: Value = serde_json::from_str(&attributes["langfuse.observation.input"]).unwrap();
        assert_eq!(
            input,
            json!([
                { "role": "system", "content": "You are terse." },
                { "role": "user", "content": "Ready?" },
            ])
        );
        let output: Value =
            serde_json::from_str(&attributes["langfuse.observation.output"]).unwrap();
        assert_eq!(output, json!({ "role": "assistant", "content": "Ready." }));
        let usage: Value =
            serde_json::from_str(&attributes["langfuse.observation.usage_details"]).unwrap();
        assert_eq!(
            usage,
            json!({
                "input": 15,
                "output": 5,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 3,
            })
        );
        assert_eq!(
            attributes["langfuse.observation.metadata.stop_reason"],
            "EndTurn"
        );
        assert_eq!(
            attributes["langfuse.observation.metadata.provider_latency_ms"],
            "700"
        );
    }

    #[test]
    fn a_failed_call_keeps_its_request_and_records_the_error() {
        let err = LlmError::Throttled("slow down".into());
        let attributes =
            exported_attributes(|span| record_generation_error(span, &request(), &err));

        assert_eq!(attributes["langfuse.observation.type"], "generation");
        assert!(attributes.contains_key("langfuse.observation.input"));
        assert_eq!(attributes["langfuse.observation.level"], "ERROR");
        assert_eq!(
            attributes["langfuse.observation.status_message"],
            "throttled: slow down"
        );
        assert!(!attributes.contains_key("langfuse.observation.output"));
    }

    #[test]
    fn the_first_token_time_is_an_iso_8601_timestamp() {
        let at = SystemTime::UNIX_EPOCH + Duration::from_millis(1_790_000_000_123);
        let attributes = exported_attributes(|span| record_completion_start(span, at));

        assert_eq!(
            attributes["langfuse.observation.completion_start_time"],
            "2026-09-21T14:13:20.123Z"
        );
    }
}
