# 0007 · Scores through the public API, as a decorator assembled by a builder

- Date: 2026-10-07
- Status: accepted; the endpoint of option 2 superseded by 0010

## Context

Step 5.12 attaches a first score, `smoke_ok`, to every model call. Scores do not travel over
OpenTelemetry: they are a separate call to the Langfuse public API. Recording a score is
plumbing, like tracing, while deciding its value is a rule of the product. The harness code
should contain neither the plumbing of tracing nor that of scoring.

## Options considered

1. `langfuse-ergonomic` 0.7.0 for the API call: its scores go through the legacy ingestion
   endpoint, which the Langfuse v4 guide says not to use for scores, and its "binary" score is a
   numeric 0 or 1, not a boolean.
2. A direct `POST /api/public/scores` with `reqwest`, behind a `Scores` trait of ours.
3. Scores sent from each command (`hello`, `chat`, later the pipeline steps).
4. A scoring decorator around `LlmClient` that applies evaluators, domain functions from
   `(LlmRequest, LlmResponse)` to a verdict, and sends the verdicts to the current trace.
5. Decorators composed by hand, or one `LlmClientBuilder` that starts from the provider's client
   and switches features on (`with_tracing`, `with_scoring`).

## Decision

Options 2, 4 and 5. The direct call is a few lines and follows the current API; it is tested
with contract tests against the real Langfuse, not with a fake of its HTTP API, because we mock
only what we own. The decorator keeps scoring out of the commands: `hello` and `chat` never name
a score, and `smoke_ok` stays in `app` as a plain, tested function. The builder fixes the order
of the decorators whatever order the features are asked in: scoring outside tracing, so that the
time spent sending a score is not counted as the model call's latency. It lives in
`observability`, the only crate that knows both features, because `llm-core` must not depend
on it.

## Consequences

A second evaluator is one more function passed to `with_scoring`. A score that cannot be sent
is a warning, never a failed call; the score is awaited before the reply returns, which costs a
CLI about 200 ms per call and could move to the background in a server. A new score takes from
10 to 60 seconds to become readable in Langfuse, and `GET /api/public/v2/scores/{id}` answers
`410` to organisations created after 2026-09-16: reads go through `/api/public/v3/scores`.
Scores that do not come from a model call, such as user feedback (Module 14), still need an
explicit call to `Scores`. When Module 10 adds features that are not observability, such as
fallback, the builder moves to a crate that depends on all of them, probably `harness`.
