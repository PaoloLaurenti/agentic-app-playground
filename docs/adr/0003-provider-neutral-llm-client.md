# 0003 · A provider-neutral LlmClient between the harness and Bedrock

- Date: 2026-09-26
- Status: accepted

## Context

The harness needs to call an LLM, and today the only provider is Bedrock through the Converse
API (ADR 0002). Converse already hides the differences between model vendors, but its SDK types
are awkward to build on: `i32` token counts, optional fields the API always fills, fifteen kinds
of content block, three near-identical error enums, builder names that change between versions.
The SDK also adds about 2,000 lines to `Cargo.lock`. From Module 7 the harness grows logic of its
own (agent loop, tools, pipeline steps, regeneration) that must be tested on every pull request
without credentials.

## Options considered

1. Use the SDK types directly everywhere and mock Bedrock at the HTTP level in tests.
2. Define neutral types and an `LlmClient` trait in `llm-core`, implemented by `llm-bedrock`,
   with `llm-core` free of any AWS dependency.

## Decision

Option 2. The benefits, in order of weight for this project:

- the harness is tested against a fake `LlmClient`: no network, no credentials, deterministic;
- behaviour can wrap a client without the harness noticing: routing and fallback (Module 10),
  rate limiting, caching;
- the SDK stays inside one crate, and harness crates build and test without it;
- the neutral types carry decisions, such as `LlmError` grouping failures by what the caller can
  do (step 4.4) instead of copying the AWS exceptions;
- a second provider, such as Claude Platform on AWS (Module 16), fits without touching the
  harness. This is the most hypothetical benefit today.

## Consequences

The runtime cost is negligible, but every feature is written twice, as a neutral type and as its
translation: tools, structured output, images. Bedrock-only features such as Guardrails have no
neutral home and go through `extra` or stay unused. The neutral types were designed with a single
provider in view and closely follow Converse, so they may prove less neutral than they look.
Revisit in Module 10, when the router is the first real user of the abstraction beyond tests, and
whenever a second provider is added. In a product without logic to test apart from the provider,
calling Converse directly would be a legitimate choice.
