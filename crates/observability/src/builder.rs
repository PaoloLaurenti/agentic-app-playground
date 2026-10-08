//! One place to build the client the application talks to: start from a provider's client and
//! switch on the features around it.

use std::sync::Arc;

use llm_core::LlmClient;

use crate::scores::Scores;
use crate::scoring::{Evaluator, ScoringClient};
use crate::traced_client::TracedClient;

/// Builds an [`LlmClient`] from a provider's client and the features switched on with `with_*`.
pub struct LlmClientBuilder {
    inner: Box<dyn LlmClient>,
    tracing: bool,
    scoring: Option<(Arc<dyn Scores>, Vec<Evaluator>)>,
}

impl LlmClientBuilder {
    pub fn new(inner: impl LlmClient + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            tracing: false,
            scoring: None,
        }
    }

    /// One span per call, which Langfuse shows as a generation.
    pub fn with_tracing(mut self) -> Self {
        self.tracing = true;
        self
    }

    /// After every completed call, each evaluator judges it and its verdict becomes a score on the
    /// current trace.
    pub fn with_scoring(
        mut self,
        scores: Arc<dyn Scores>,
        evaluators: impl IntoIterator<Item = Evaluator>,
    ) -> Self {
        self.scoring = Some((scores, evaluators.into_iter().collect()));
        self
    }

    pub fn build(self) -> Box<dyn LlmClient> {
        let mut client = self.inner;
        if self.tracing {
            client = Box::new(TracedClient::around(client));
        }
        if let Some((scores, evaluators)) = self.scoring {
            client = Box::new(ScoringClient::around(client, scores, evaluators));
        }
        client
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt;
    use llm_core::{ContentBlock, LlmRequest, LlmResponse};

    use super::*;
    use crate::Verdict;
    use crate::test_support::{FakeClient, RecordedScores, block_on, exported_spans, request};

    fn always_ok(_req: &LlmRequest, _resp: &LlmResponse) -> Verdict {
        Verdict {
            name: "always_ok",
            value: true,
        }
    }

    #[test]
    fn with_tracing_every_call_is_a_call_model_span() {
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_tracing()
            .build();

        let spans = exported_spans(|| {
            block_on(client.complete(request())).unwrap();
        });

        let names: Vec<_> = spans.iter().map(|span| span.name.as_str()).collect();
        assert_eq!(names, ["call-model"]);
    }

    fn request_with_effort() -> LlmRequest {
        LlmRequest {
            extra: serde_json::json!({ "effort": "low", "thinking": "adaptive" }),
            ..request()
        }
    }

    fn model_parameters(span: &crate::test_support::ExportedSpan) -> serde_json::Value {
        serde_json::from_str(&span.attributes["langfuse.observation.model.parameters"]).unwrap()
    }

    #[test]
    fn with_tracing_the_generation_shows_the_parameters_the_provider_sends() {
        let client = LlmClientBuilder::new(FakeClient {
            drops: Some("effort"),
            ..FakeClient::default()
        })
        .with_tracing()
        .build();

        let spans = exported_spans(|| {
            block_on(client.complete(request_with_effort())).unwrap();
        });

        assert_eq!(
            model_parameters(&spans[0]),
            serde_json::json!({ "max_tokens": 50, "thinking": "adaptive" })
        );
    }

    #[test]
    fn with_tracing_a_streamed_generation_shows_the_parameters_the_provider_sends() {
        let client = LlmClientBuilder::new(FakeClient {
            drops: Some("effort"),
            ..FakeClient::default()
        })
        .with_tracing()
        .build();

        let spans = exported_spans(|| {
            block_on(async {
                let events = client.stream(request_with_effort()).await.unwrap();
                events.collect::<Vec<_>>().await;
            });
        });

        assert_eq!(
            model_parameters(&spans[0]),
            serde_json::json!({ "max_tokens": 50, "thinking": "adaptive" })
        );
    }

    #[test]
    fn with_tracing_a_failed_generation_shows_the_parameters_the_provider_sends() {
        let client = LlmClientBuilder::new(FakeClient {
            fail: true,
            drops: Some("effort"),
        })
        .with_tracing()
        .build();

        let spans = exported_spans(|| {
            block_on(client.complete(request_with_effort())).unwrap_err();
        });

        assert_eq!(
            model_parameters(&spans[0]),
            serde_json::json!({ "max_tokens": 50, "thinking": "adaptive" })
        );
    }

    #[test]
    fn the_built_client_prepares_a_request_as_the_provider_would() {
        let client = LlmClientBuilder::new(FakeClient {
            drops: Some("effort"),
            ..FakeClient::default()
        })
        .with_tracing()
        .with_scoring(
            Arc::new(RecordedScores::default()),
            [always_ok as Evaluator],
        )
        .build();

        let prepared = client.prepare(request_with_effort());

        assert_eq!(
            prepared.extra,
            serde_json::json!({ "thinking": "adaptive" })
        );
    }

    #[test]
    fn with_tracing_the_generation_names_the_prompt_and_version_it_used() {
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_tracing()
            .build();
        let request = LlmRequest {
            prompt: Some(llm_core::PromptRef {
                name: "safety-classifier".into(),
                version: 2,
            }),
            ..request()
        };

        let spans = exported_spans(|| {
            block_on(client.complete(request)).unwrap();
        });

        let attributes = &spans[0].attributes;
        assert_eq!(
            attributes["langfuse.observation.prompt.name"],
            "safety-classifier"
        );
        assert_eq!(attributes["langfuse.observation.prompt.version"], "2");
    }

    #[test]
    fn with_tracing_a_generation_without_a_prompt_names_none() {
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_tracing()
            .build();

        let spans = exported_spans(|| {
            block_on(client.complete(request())).unwrap();
        });

        let attributes = &spans[0].attributes;
        assert!(!attributes.contains_key("langfuse.observation.prompt.name"));
        assert!(!attributes.contains_key("langfuse.observation.prompt.version"));
    }

    #[test]
    fn with_scoring_every_verdict_is_sent_to_the_current_trace() {
        let scores = Arc::new(RecordedScores::default());
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_scoring(scores.clone(), [always_ok as Evaluator])
            .build();

        let spans = exported_spans(|| {
            let _root = tracing::info_span!("hello").entered();
            block_on(client.complete(request())).unwrap();
        });

        let names: Vec<_> = spans.iter().map(|span| span.name.as_str()).collect();
        assert_eq!(names, ["hello"]);
        assert_eq!(
            *scores.sent.lock().unwrap(),
            [(spans[0].trace_id.clone(), "always_ok".to_owned(), true)]
        );
    }

    #[test]
    fn sending_the_scores_is_not_counted_in_the_model_calls_time() {
        let scores = Arc::new(RecordedScores {
            delay: Duration::from_millis(200),
            ..RecordedScores::default()
        });
        // Asked in the other order on purpose: `build` decides the order.
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_scoring(scores.clone(), [always_ok as Evaluator])
            .with_tracing()
            .build();

        let spans = exported_spans(|| {
            let _root = tracing::info_span!("hello").entered();
            block_on(client.complete(request())).unwrap();
        });

        let call_model = spans.iter().find(|span| span.name == "call-model").unwrap();
        assert!(
            call_model.duration < Duration::from_millis(200),
            "{:?}",
            call_model.duration
        );
        assert_eq!(scores.sent.lock().unwrap()[0].0, call_model.trace_id);
    }

    fn reply_is_ready(_req: &LlmRequest, resp: &LlmResponse) -> Verdict {
        Verdict {
            name: "reply_is_ready",
            value: resp.message.content == [ContentBlock::Text("Ready.".into())],
        }
    }

    #[test]
    fn a_stream_is_judged_on_the_whole_reply_once_it_ends() {
        let scores = Arc::new(RecordedScores::default());
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_scoring(scores.clone(), [reply_is_ready as Evaluator])
            .build();

        exported_spans(|| {
            let _root = tracing::info_span!("hello").entered();
            let events: Vec<_> =
                block_on(async { client.stream(request()).await.unwrap().collect().await });
            assert_eq!(events.len(), 4);
        });

        let sent = scores.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!((sent[0].1.as_str(), sent[0].2), ("reply_is_ready", true));
    }

    #[test]
    fn a_score_that_cannot_be_sent_is_a_warning_not_a_failed_call() {
        let scores = Arc::new(RecordedScores {
            fail: true,
            ..RecordedScores::default()
        });
        let client = LlmClientBuilder::new(FakeClient::default())
            .with_scoring(scores, [always_ok as Evaluator])
            .build();

        let spans = exported_spans(|| {
            let _root = tracing::info_span!("hello").entered();
            assert!(block_on(client.complete(request())).is_ok());
        });

        assert_eq!(spans[0].events, ["the score could not be sent"]);
    }

    #[test]
    fn a_failed_call_is_not_judged() {
        let scores = Arc::new(RecordedScores::default());
        let client = LlmClientBuilder::new(FakeClient {
            fail: true,
            ..FakeClient::default()
        })
        .with_scoring(scores.clone(), [always_ok as Evaluator])
        .build();

        exported_spans(|| {
            let _root = tracing::info_span!("hello").entered();
            assert!(block_on(client.complete(request())).is_err());
        });

        assert!(scores.sent.lock().unwrap().is_empty());
    }
}
