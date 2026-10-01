# 0005 · Export to Langfuse with the plain OTLP exporter

- Date: 2026-10-01
- Status: accepted

## Context

Spans reach Langfuse through its OpenTelemetry endpoint (ADR 0004), and `tracing-opentelemetry`
turns `tracing` spans into OpenTelemetry ones. Every OpenTelemetry crate in the chain must share
one version of `opentelemetry`, or the layer's spans never reach the exporter. On 2026-10-01
`tracing-opentelemetry` 0.34 requires `opentelemetry` 0.33.

## Options considered

1. `opentelemetry-langfuse` 0.6.1, a builder preconfigured for Langfuse: it pins
   `opentelemetry` 0.31 and was last released in December 2025.
2. `opentelemetry-otlp` 0.33 configured by hand: endpoint, two headers, HTTP with protobuf.

## Decision

Option 2. Option 1 cannot be combined with the current `tracing-opentelemetry`, and what it adds
is about fifteen lines in `crates/observability`. The exporter uses the blocking `reqwest`
client with rustls, which runs on the batch processor's own thread, so shutdown is a plain
synchronous flush.

## Consequences

One dependency less to wait for when OpenTelemetry releases a new version. The Langfuse
specifics now live in our code: the `/api/public/otel/v1/traces` path, Basic authentication and
the `x-langfuse-ingestion-version: 4` header, without which spans can take up to ten minutes to
appear. Revisit if `opentelemetry-langfuse` catches up and adds something worth having.
