//! Judging model calls: evaluators are domain rules that look at a call and give a verdict, and the
//! verdicts become scores on the call's trace.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream::BoxStream;
use llm_core::{LlmClient, LlmError, LlmEvent, LlmRequest, LlmResponse};

use crate::reply_so_far::ReplySoFar;
use crate::scores::{Scores, current_trace_id};

/// A judgment on one model call: a name, such as `smoke_ok`, and whether the call passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub name: &'static str,
    pub value: bool,
}

/// A rule that judges a completed call from its request and its response.
pub type Evaluator = fn(&LlmRequest, &LlmResponse) -> Verdict;

/// A decorator that judges every completed call and sends the verdicts as scores on the current
/// trace. The client inside knows nothing about scores.
pub(crate) struct ScoringClient {
    inner: Box<dyn LlmClient>,
    judge: Judge,
}

/// The evaluators and where their verdicts go; a stream carries its own copy until it ends.
#[derive(Clone)]
struct Judge {
    scores: Arc<dyn Scores>,
    evaluators: Vec<Evaluator>,
}

impl Judge {
    /// Judges a completed call and sends every verdict to the trace, if there is one.
    async fn judge(&self, trace_id: Option<&str>, req: &LlmRequest, resp: &LlmResponse) {
        let Some(trace_id) = trace_id else {
            return;
        };
        for evaluate in &self.evaluators {
            let verdict = evaluate(req, resp);
            // A lost score must not cost the user the reply.
            if let Err(err) = self
                .scores
                .boolean(trace_id, verdict.name, verdict.value)
                .await
            {
                tracing::warn!(score = verdict.name, error = %err, "the score could not be sent");
            }
        }
    }
}

impl ScoringClient {
    pub(crate) fn around(
        inner: Box<dyn LlmClient>,
        scores: Arc<dyn Scores>,
        evaluators: Vec<Evaluator>,
    ) -> Self {
        Self {
            inner,
            judge: Judge { scores, evaluators },
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for ScoringClient {
    fn prepare(&self, req: LlmRequest) -> LlmRequest {
        self.inner.prepare(req)
    }

    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let response = self.inner.complete(req.clone()).await?;
        self.judge
            .judge(current_trace_id().as_deref(), &req, &response)
            .await;
        Ok(response)
    }

    async fn stream(
        &self,
        req: LlmRequest,
    ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
        // The trace is the caller's, so it is read now, while the caller's span is current.
        let trace_id = current_trace_id();
        let events = self.inner.stream(req.clone()).await?;
        let state = (
            events,
            ReplySoFar::default(),
            self.judge.clone(),
            req,
            trace_id,
        );
        // Every event passes through unchanged; the reply is judged once the stream ends.
        let events = futures::stream::unfold(Some(state), |state| async move {
            let (mut events, mut so_far, judge, req, trace_id) = state?;
            match events.next().await {
                Some(Ok(event)) => {
                    so_far.add(&event);
                    Some((Ok(event), Some((events, so_far, judge, req, trace_id))))
                }
                Some(Err(err)) => Some((Err(err), None)),
                None => {
                    if let Some(response) = so_far.into_response() {
                        judge.judge(trace_id.as_deref(), &req, &response).await;
                    }
                    None
                }
            }
        });
        Ok(events.boxed())
    }
}
