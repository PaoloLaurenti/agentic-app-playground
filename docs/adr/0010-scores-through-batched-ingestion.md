# 0010 · Scores through batched ingestion

- Date: 2026-10-09
- Status: accepted

## Context

ADR 0007 sends each score with `POST /api/public/scores`, and rejected the ingestion endpoint
because the Langfuse v4 guide then said not to use it for scores. In EXP-004, the first run of
40 cases, Langfuse rejected 7 of the 40 `smoke_ok` scores with `429 Too Many Requests`.

Langfuse Cloud limits requests per organization, in groups. `POST /api/public/scores` falls under
the General API group, "all other public APIs", which the Hobby plan limits to 30 requests a
minute, shared with every other call of the organization. Batched `/ingestion` and `/otel` form
the Tracing group, 1,000 requests a minute on Hobby. A `429` carries a `Retry-After` header.

The Langfuse sources disagree on which endpoint to use for scores. The OpenAPI specification
says "To write scores, prefer `POST /api/public/scores`". The deprecated API migration guide
says the current Python and JS SDKs "intentionally batch score writes as `score-create` events
through `POST /ingestion`", and that Langfuse v4 continues to accept that event type after
2026-11-16, when the endpoint starts rejecting trace and observation events.

Module 11 sends far more scores than one run of 6.10: cases × repetitions × evaluators.

## Options considered

1. Stay on `POST /api/public/scores` and retry after a `429`, waiting for `Retry-After`. It
   follows the specification's preference, but every run is held to 30 scores a minute, less
   whatever else the organization calls.
2. Send each score as a `score-create` event to `POST /api/public/ingestion`, the route the
   official SDKs take.
3. Drop scoring from the evaluation runner, whose metrics do not depend on it. The limit comes
   back with the scores of Module 11.

## Decision

Option 2. It is the route Langfuse's own SDKs depend on, it stays open after the v4 cutover, and
it sits in a group with a much higher limit. The page of limits does not name which endpoints
its separate "Deprecated tracing" group (100 requests a minute) covers; even there, the limit is
above today's 30. `ScoreClient` keeps the `Scores` trait of ADR 0007, so nothing outside it
changes. The rest of ADR 0007 stands.

## Consequences

The endpoint answers `207 Multi-Status` with lists of `successes` and `errors` instead of a `4xx`
for a rejected event, so `ScoreClient` reads the response body to tell whether its score was
accepted; the contract tests check it against the real Langfuse. The client chooses the score's
id, since the event body carries it, instead of reading it back. One request still carries one
score, because scoring awaits each score after its call: batching several scores per request is
possible if a run ever needs it. If Langfuse ever moves the SDKs' scores elsewhere, this decision
must be revisited.
