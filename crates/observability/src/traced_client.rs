//! A decorator that observes any [`LlmClient`]: one span per call, which Langfuse shows as a
//! generation. The client inside only calls its provider and knows nothing about tracing.

use std::time::SystemTime;

use futures::StreamExt;
use futures::stream::BoxStream;
use llm_core::{LlmClient, LlmError, LlmEvent, LlmRequest, LlmResponse};
use tracing::{Instrument, Span, field};

use crate::generation::{record_completion_start, record_generation, record_generation_error};
use crate::reply_so_far::ReplySoFar;

/// Wraps a client and behaves exactly like it, recording every call on the way.
pub(crate) struct TracedClient {
    inner: Box<dyn LlmClient>,
}

impl TracedClient {
    #[cfg(test)]
    pub(crate) fn new(inner: impl LlmClient + 'static) -> Self {
        Self::around(Box::new(inner))
    }

    pub(crate) fn around(inner: Box<dyn LlmClient>) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl LlmClient for TracedClient {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let span = call_span(&req, "complete");
        let result = self
            .inner
            .complete(req.clone())
            .instrument(span.clone())
            .await;
        match &result {
            Ok(response) => {
                record_log_fields(&span, response);
                record_generation(&span, &req, response);
            }
            Err(err) => record_failure(&span, &req, err),
        }
        result
    }

    async fn stream(
        &self,
        req: LlmRequest,
    ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
        let span = call_span(&req, "stream");
        let events = match self
            .inner
            .stream(req.clone())
            .instrument(span.clone())
            .await
        {
            Ok(events) => events,
            Err(err) => {
                record_failure(&span, &req, &err);
                return Err(err);
            }
        };

        // The span travels in the state, so it stays open until the stream ends, and the state
        // gathers the reply, which becomes the generation's output at the end. The state becomes
        // `None` after an error, which ends the stream.
        let state = TracedStream {
            events,
            span,
            req,
            so_far: ReplySoFar::default(),
        };
        let events = futures::stream::unfold(Some(state), |state| async move {
            let mut state = state?;
            let next = state.events.next().instrument(state.span.clone()).await;
            match next {
                Some(Ok(event)) => {
                    if state.so_far.add(&event) {
                        record_completion_start(&state.span, SystemTime::now());
                    }
                    Some((Ok(event), Some(state)))
                }
                Some(Err(err)) => {
                    record_failure(&state.span, &state.req, &err);
                    Some((Err(err), None))
                }
                None => {
                    state.finish();
                    None
                }
            }
        });
        Ok(events.boxed())
    }
}

// The span's own fields are what the terminal log prints when the span closes. They are declared
// empty and recorded when the response arrives.

fn call_span(req: &LlmRequest, operation: &'static str) -> Span {
    tracing::info_span!(
        "call-model",
        model = req.model.as_str(),
        operation,
        input_tokens = field::Empty,
        output_tokens = field::Empty,
        cache_read_tokens = field::Empty,
        cache_write_tokens = field::Empty,
        latency_ms = field::Empty,
        stop_reason = field::Empty,
    )
}

fn record_log_fields(span: &Span, response: &LlmResponse) {
    let usage = &response.usage;
    span.record("input_tokens", usage.input_tokens);
    span.record("output_tokens", usage.output_tokens);
    span.record("cache_read_tokens", usage.cache_read_tokens);
    span.record("cache_write_tokens", usage.cache_write_tokens);
    span.record(
        "latency_ms",
        u64::try_from(response.latency.as_millis()).unwrap_or(u64::MAX),
    );
    span.record("stop_reason", field::debug(&response.stop_reason));
}

/// The error's text starts with its kind (`throttled:`, `invalid request:`…), so the kinds are
/// told apart in the logs.
fn record_failure(span: &Span, req: &LlmRequest, err: &LlmError) {
    span.in_scope(|| tracing::error!(error = %err, "the call failed"));
    record_generation_error(span, req, err);
}

/// What a stream carries from one event to the next.
struct TracedStream {
    events: BoxStream<'static, Result<LlmEvent, LlmError>>,
    span: Span,
    req: LlmRequest,
    so_far: ReplySoFar,
}

impl TracedStream {
    /// Records the whole generation once the stream is over.
    fn finish(self) {
        match self.so_far.into_response() {
            Some(response) => {
                record_log_fields(&self.span, &response);
                record_generation(&self.span, &self.req, &response);
            }
            None => record_failure(
                &self.span,
                &self.req,
                &LlmError::Other("the stream ended without a stop reason or token counts".into()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{FakeClient, block_on, exported_spans, request};

    #[test]
    fn a_completed_call_is_one_generation_named_call_model() {
        let client = TracedClient::new(FakeClient { fail: false });

        let spans = exported_spans(|| {
            block_on(client.complete(request())).unwrap();
        });

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "call-model");
        assert_eq!(
            spans[0].attributes["langfuse.observation.type"],
            "generation"
        );
        assert!(
            spans[0]
                .attributes
                .contains_key("langfuse.observation.output")
        );
    }

    #[test]
    fn a_failed_call_is_still_returned_and_recorded_as_an_error() {
        let client = TracedClient::new(FakeClient { fail: true });

        let spans = exported_spans(|| {
            let result = block_on(client.complete(request()));
            assert!(matches!(result, Err(LlmError::Throttled(_))));
        });

        assert_eq!(spans[0].attributes["langfuse.observation.level"], "ERROR");
    }

    #[test]
    fn a_stream_passes_every_event_through_and_records_the_whole_reply() {
        let client = TracedClient::new(FakeClient { fail: false });

        let spans = exported_spans(|| {
            let events: Vec<_> =
                block_on(async { client.stream(request()).await.unwrap().collect().await });
            assert_eq!(events.len(), 4);
        });

        let attributes = &spans[0].attributes;
        let output: serde_json::Value =
            serde_json::from_str(&attributes["langfuse.observation.output"]).unwrap();
        assert_eq!(output["content"], "Ready.");
        assert!(attributes.contains_key("langfuse.observation.completion_start_time"));
    }
}
