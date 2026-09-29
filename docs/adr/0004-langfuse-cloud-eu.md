# 0004 · Langfuse Cloud in the EU region, no self-hosting

- Date: 2026-09-28
- Status: accepted

## Context

From Module 5 every LLM call is exported to Langfuse as a trace, with full prompts and replies.
The domain is healthcare, so observability data must stay in the EU like the model calls
(`AGENTS.md`, ADR 0002). The data is synthetic and the project has one user. The aim is to learn
tracing, prompt management and evaluation, not to run infrastructure.

## Options considered

1. Langfuse Cloud, EU region (`https://cloud.langfuse.com`): data in AWS `eu-west-1`, the same
   region as Bedrock, and no infrastructure to run.
2. Self-hosted with Docker Compose, locally or later on AWS EU: `web` and `worker`, PostgreSQL,
   ClickHouse, Redis/Valkey and S3/MinIO, all to install, upgrade and back up.
3. LangSmith, from LangChain, with an EU region on GCP (`eu.smith.langchain.com`): closed
   source, and its extra over Langfuse is managed deployment of LangGraph agents.

## Decision

Option 1, for the whole tutorial. It meets data residency with nothing to operate. Self-hosting
teaches the Langfuse architecture, not agentic applications, and is not pursued. LangSmith's
tracing, evaluation and prompt management match what the tutorial uses from Langfuse, and its
managed deployment does not apply to a hand-written Rust harness deployed on AWS. The free Hobby
plan is enough: 50,000 units a month, where a unit is a trace, an observation or a score, 30
days of data access and 2 users (langfuse.com/pricing, 2026-09-28).

## Consequences

The project needs only an API key pair in `.env` and the EU host. Traces older than 30 days are
no longer visible, so every result worth keeping goes into the Work Log and the experiment cards,
as it already does. Evaluation runs with many repetitions (Module 11) consume units quickly:
check usage before large runs. Revisit before production (Module 15), where Langfuse Cloud must
be reviewed with legal as a sub-processor, and whenever real personal data is in view.
