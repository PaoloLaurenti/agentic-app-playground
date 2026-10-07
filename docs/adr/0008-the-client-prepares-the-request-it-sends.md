# 0008 · The client prepares the request it sends

- Date: 2026-10-07
- Status: accepted

## Context

Step 6.4 takes a parameter such as effort from a prompt's `config` for the first time, and Claude
Haiku 4.5 rejects effort: Bedrock answers "This model does not support the effort parameter".
A capability table in `llm-bedrock` must drop what the target model does not accept. The
generation in Langfuse must show the parameters actually sent, so that an experiment never
believes it changed a variable that was dropped. `TracedClient` wraps `BedrockClient` and
records the request as it receives it, so a parameter dropped inside `BedrockClient` alone
would still show up in the trace.

## Options considered

1. A filter decorator in `llm-bedrock`, wrapped by the app around the client that
   `LlmClientBuilder` builds, outside tracing.
2. The same decorator inside the builder, which receives the filter as a function
   (`with_request_filter`), so that `observability` does not depend on `llm-bedrock`.
3. The table inside `BedrockClient`, with `LlmResponse` and `LlmEvent::Metadata` carrying the
   parameters actually sent back to `TracedClient`.
4. A `prepare` method on `LlmClient` that returns the request as the client will send it.
   `BedrockClient` applies the table there and in its own calls. Decorators forward it, and
   `TracedClient` records the prepared request.

## Decision

Option 4, with no default implementation of `prepare`. The table stays in the provider, which
is the one that knows its models. The trace shows the parameters actually sent, failed calls
included. Neither the app nor the builder has to remember a filter. A default would return the
request unchanged, so a new decorator that forgot to forward `prepare` would silently break the
chain. Without a default, every client has to decide, and the compiler enforces it. Options 1
and 2 put a piece of a provider's knowledge in the app or in the builder. Option 3 changes
three crates and loses the parameters when a call fails.

## Consequences

Every `LlmClient` writes one more method: a decorator forwards it to its inner client, a fake
returns the request. `BedrockClient` prepares a request even when no decorator wraps it, so
under tracing it prepares the request twice. The second pass finds nothing to drop and logs
nothing. The table has one row, filled in when a model rejected a parameter we sent.
`temperature` arrives when `LlmRequest` first carries it (6.11). In Module 10 the table moves
into the `ModelRegistry`, and `prepare` is where the router would apply it.
