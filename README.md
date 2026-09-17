# agentic-app-playground

A playground for learning how to build an **agentic application**: a hand-written Rust harness
driving an LLM on Amazon Bedrock, observed and evaluated with Langfuse.

This is not a product. It is a learning path with a realistic guiding project.

## Where to start

**[`TUTORIAL.md`](TUTORIAL.md)** is the path: 16 modules in order, each with checkboxes, a
completion criterion and a self-check question. Progress is tracked in the index at section 0.2.

| Modules | Topic |
|---|---|
| 1–4 | Foundations, Rust setup, enabling Bedrock, first call from code |
| 5–6 | Langfuse: tracing, and prompt management with versions and labels |
| 7–9 | Tool calling and the agent loop, context engineering, the full pipeline |
| 10–12 | Model router, evaluation with datasets and experiments, regression testing |
| 13–15 | Security and privacy, production observability, deployment |
| 16 | Optional deep dives |

## The guiding project

A conversational assistant for a healthcare service. Every user message goes through a pipeline
with three blocking LLM calls and one asynchronous call:

```text
STATE_READ → SAFETY → [RETRIEVAL] → GENERATION → GUARDRAILS → STATE_WRITE → reply
                                                                              ↓
                                                                        PROFILE_UPDATE
```

`SAFETY` is a gate: if it detects acute risk, the pipeline stops and returns a fixed escalation
text. `GUARDRAILS` validates the reply and, when needed, triggers a regeneration.

> ⚠️ The domain is healthcare but this remains a playground: **synthetic data only**, never real
> patient data.

## Prerequisites

- Stable Rust (the version is pinned in `.tool-versions`, managed with asdf)
- AWS CLI v2 and an AWS account with access to Amazon Bedrock in an EU region
- A Langfuse account (EU cloud region) or Docker for local self-hosting

## Getting started

```bash
cp .env.example .env    # then fill in the values by following Modules 3 and 5
rustc --version         # check that the pinned toolchain is active
```

Then open `TUTORIAL.md` and start at Module 1. The Cargo workspace and the `just` commands are
created in Module 2, so until then there is no code to build.

## Conventions

They live in **[`AGENTS.md`](AGENTS.md)**: non-negotiable constraints, stack, crate layout, and
the rules for commits, prompts, tests and decisions. Architecture decisions go in `docs/adr/`,
starting from the `docs/adr/0000-template.md` template.
