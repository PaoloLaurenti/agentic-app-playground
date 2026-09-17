# Agent instructions

## What this repository is

A learning playground for building an **agentic application**: a hand-written harness driving
an LLM. It is not a product and will not go to production as it stands.

The path is in **`TUTORIAL.md`**: 16 modules with checkboxes. It is the map of this project.

> **Before proposing or starting any work:** read the progress section in `TUTORIAL.md` §0.2 and
> resume from the first module that is not complete. The modules are cumulative, so do not skip
> ahead. When a module is finished, tick its checkbox in the index along with its steps.

## The guiding project

A conversational assistant for a healthcare service. For every user message:

`STATE_READ` (database) → `SAFETY` (LLM, blocking gate) → `RETRIEVAL` (vector search, in parallel)
→ `GENERATION` (LLM, structured output) → `GUARDRAILS` (LLM, with regeneration)
→ `STATE_WRITE` (database) → reply → `PROFILE_UPDATE` (LLM, asynchronous)

Inspired by an internal Mia Clinic document, taken as a starting point rather than a specification.

## Non-negotiable constraints

- **Synthetic data only.** The domain is healthcare, but real patient data never enters the
  repository, not even anonymized. This covers datasets, prompts, examples, tests and traces.
- **EU only.** Bedrock exclusively through `eu.*` inference profiles: never `us.*`, never
  `global.*`. The same applies to observability: Langfuse Cloud EU region, or self-hosted in the EU.
- **No secrets in the repository.** `.env` is git-ignored; `.env.example` documents the variables
  without values. Keys and credentials never land in code, tests or commits.
- **No agent framework in the core.** The harness is written by hand: that is the thing being
  learned. Rig is an optional comparison (Module 16), not a dependency of the path.
- **One change at a time.** When measuring anything (prompt, model, effort), change a single
  variable and record the result. This is a working rule, not just a methodological one.

## Stack

| Area | Choice |
|---|---|
| Language | Rust, edition 2024, `tokio` runtime. Toolchain pinned in `.tool-versions` (asdf). |
| LLM provider | Amazon Bedrock, **Converse API**, official AWS SDK for Rust (`aws-sdk-bedrockruntime`). |
| Region | `eu-west-1` or `eu-central-1`, `eu.*` inference profiles. |
| LLM engineering platform | **Langfuse**: tracing, prompt management, datasets and experiments, dashboards and alerts. |
| Langfuse connection | OpenTelemetry (`/api/public/otel`) for traces, the public REST API for prompts, datasets and scores. |
| Persistence | SQLite locally, PostgreSQL with `pgvector` when vector search is needed. |
| External tools | Model Context Protocol through the official `rmcp` crate. |

## Layout

Current:

```text
TUTORIAL.md        the learning path, 16 modules
AGENTS.md          this file
docs/adr/          architecture decisions, from template 0000
```

Planned, created in Module 2 (do not build it ahead of time):

```text
crates/llm-core/       neutral types and the LlmClient trait
crates/llm-bedrock/    implementation over Converse
crates/observability/  tracing, OpenTelemetry, Langfuse client
crates/prompts/        prompt loading from Langfuse with cache and fallback
crates/harness/        agent loop, tool registry, context, pipeline steps
crates/router/         model registry, routing, fallback
crates/evals/          evaluation runner and dataset runs
crates/app/            binary: CLI and HTTP server
prompts/               prompt snapshots downloaded from Langfuse (fallback and PR diffs)
datasets/              test cases in JSONL, synthetic only
```

## Conventions

### Language

**Everything written into this repository is in English**: documentation, code, comments,
prompts, datasets, commit messages and pull requests. The only exception is content that is
data about the product's Italian-speaking users, such as the example user messages in the
safety dataset, which stay in Italian because that is what the classifier must handle.

**Conversation with the repository owner happens in Italian.**

### Commit messages

- In **English**, **imperative** mood: `Add`, `Fix`, `Rewrite`, `Move`. Never `Added`, `Adds`.
- **At most 2 lines**: a subject, plus at most one body line separated by a blank line.
- **No generated sign-off trailers** (`Co-Authored-By` and the like).
- One commit per understandable unit of work, not one commit per file.

### Rust code

- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` must pass before every commit.
- `llm-core` depends on neither `aws-sdk-*` nor `observability`: that boundary is what makes
  mocks, a second provider and an observability backend swap possible.
- Typed errors with `thiserror` in libraries, `anyhow` only in binaries.
- No `unwrap()` outside tests.

### Prompts

Prompts do not live inline in the code. They live in Langfuse with versions and labels
(`production`), and the downloaded snapshots live in `prompts/` so that every change shows up in
a pull request diff and acts as a fallback when Langfuse is unreachable. The prompt's name and
version are attached to every generation's attributes.

### Decisions

Every non-trivial decision becomes an ADR in `docs/adr/`, numbered sequentially, copied from
`docs/adr/0000-template.md`. Short is fine: five lines beats no ADR.

### Tests

Four levels at different cadences (details in Module 12):

1. harness unit tests, no network, on every pull request;
2. Bedrock adapter tests against simulated responses, no network, on every pull request;
3. contract tests against the real services, `#[ignore]`, nightly;
4. regression experiments over datasets with thresholds, nightly.

Never assert on the exact text of a model's output: assert properties, over several repetitions,
above a threshold set above the noise.

## Commands

The Cargo workspace does not exist yet: it is created in Module 2 together with the `justfile`.
The planned commands are `just check`, `just test`, `just eval`, `just run`.

Until then the only useful command is a toolchain check: `rustc --version`.
