# 0006 · Observe LLM calls with a decorator around LlmClient

- Date: 2026-10-01
- Status: accepted

## Context

Every model call must become a Langfuse generation: model, parameters, input, output, usage,
time to first token, error. Observability is a cross-cutting concern: the code that calls a
provider should say what it does, not also how it is watched. The first version recorded the
attributes from inside `llm-bedrock`, and its streaming code grew a second job, collecting the
reply for the trace, next to reading Bedrock's events. The tutorial (5.8) asked for a helper
wired to the span in `llm-bedrock`.

## Options considered

1. Calls to `observability` helpers inside each provider crate, as the tutorial suggested.
2. Domain events, such as "generation completed", with the recording hooked onto them.
3. A decorator: `TracedClient<C: LlmClient>` in `observability`, which implements `LlmClient`,
   opens the span, calls the client inside and records the generation from the neutral types.

## Decision

Option 3. A span has a start, an end and a parent, and the decorator gets all three for free:
it opens the span before the call and closes it after, and the span nests under whatever span
is current. An event raised after the call would have to carry its timings and its parent to
rebuild the span. The decorator only sees `LlmRequest`, `LlmResponse`, `LlmEvent` and
`LlmError`, so it works for any provider, and ADR 0003 already counted wrapping a client as a
benefit of the trait. The trace-wide attributes follow the same idea: a span processor adds
environment, release and trace name to every span, so the code that opens spans does nothing
for them.

## Consequences

`llm-bedrock` no longer depends on `tracing` or `observability`, and a second provider gets the
same generations without a line of tracing code. The application opts in where it builds the
client: `TracedClient::new(BedrockClient::new(region).await)`. A stream abandoned before its
end is exported without output or usage, because the decorator never sees the end. Domain
events remain the right tool for facts without a duration, such as a guardrail verdict that
becomes a score (Module 9) or user feedback.
