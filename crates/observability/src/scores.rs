//! Scores: judgments attached to a trace after the fact, such as a smoke check, a guardrail verdict
//! or a user's thumbs up. A score is not a span, so it does not travel over OpenTelemetry: it is a
//! `score-create` event sent to the Langfuse ingestion API, `POST /api/public/ingestion`, whose rate
//! limit is far above that of `POST /api/public/scores` (ADR 0010).

use std::time::SystemTime;

use opentelemetry::trace::TraceContextExt;
use serde_json::json;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::{Error, LangfuseConfig, basic_authorization};

/// Where scores go. The code that judges a trace depends on this trait, not on Langfuse, so its
/// tests can use a fake of their own.
#[async_trait::async_trait]
pub trait Scores: Send + Sync {
    /// Attaches a true or false judgment to a trace and returns the new score's id.
    async fn boolean(&self, trace_id: &str, name: &str, value: bool) -> Result<String, Error>;
}

/// Sends scores to one Langfuse project.
pub struct ScoreClient {
    http: reqwest::Client,
    endpoint: String,
    authorization: String,
    environment: String,
}

impl ScoreClient {
    pub fn new(config: &LangfuseConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: format!("{}/api/public/ingestion", config.host.trim_end_matches('/')),
            authorization: basic_authorization(config),
            // The score is filed under the same environment as the traces it judges.
            environment: config.environment.clone(),
        }
    }
}

#[async_trait::async_trait]
impl Scores for ScoreClient {
    async fn boolean(&self, trace_id: &str, name: &str, value: bool) -> Result<String, Error> {
        // The client chooses the score's id, since the event carries it.
        let id = uuid::Uuid::new_v4().to_string();
        // Langfuse takes a boolean as the number 1 or 0 with the `BOOLEAN` data type; without the
        // data type it would file the score as numeric.
        let body = json!({
            "batch": [{
                // The event's own id, which Langfuse uses to drop a duplicate delivery.
                "id": uuid::Uuid::new_v4().to_string(),
                "type": "score-create",
                "timestamp": humantime::format_rfc3339_millis(SystemTime::now()).to_string(),
                "body": {
                    "id": id,
                    "traceId": trace_id,
                    "name": name,
                    "value": if value { 1 } else { 0 },
                    "dataType": "BOOLEAN",
                    "environment": self.environment,
                },
            }],
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
            return Err(Error::ScoreRejected { status, reason });
        }
        Ok(id)
    }
}

/// The id of the trace the current span belongs to, as Langfuse shows it, so that a score can be
/// attached to it. `None` when spans are not exported, because then no trace exists.
pub fn current_trace_id() -> Option<String> {
    let context = tracing::Span::current().context();
    let span_context = context.span().span_context().clone();
    span_context
        .is_valid()
        .then(|| span_context.trace_id().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::exported_spans;

    #[test]
    fn inside_a_span_it_is_the_id_of_the_trace_that_span_is_exported_in() {
        let mut trace_id = None;
        let spans = exported_spans(|| {
            trace_id = tracing::info_span!("hello").in_scope(current_trace_id);
        });

        assert_eq!(trace_id, Some(spans[0].trace_id.clone()));
    }

    #[test]
    fn outside_any_span_there_is_no_trace_to_score() {
        assert_eq!(current_trace_id(), None);
    }
}
