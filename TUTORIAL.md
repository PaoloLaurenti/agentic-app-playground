# Building an agentic application in Rust on Amazon Bedrock, observed with Langfuse

> A step-by-step learning path, written for someone starting from a rough idea of what an LLM harness is.
> Every step is a checkbox: tick `[x]` when it is done. State of the art: September 2026.

---

## 0. How to use this tutorial

### 0.1 Structure

- The modules go in order. Modules **1–9** are the core (concepts, Bedrock, Langfuse, harness, prompts, context, pipeline). Modules **10–15** are the production half (router, evaluation, testing, security, observability, deployment). Module **16** collects optional deep dives.
- Every module has the same sections: **Goal**, **Why it matters**, **Steps** (the checkboxes), **Done when** (an objective completion criterion), **Self-check** (what you must be able to explain out loud), **Going deeper**.
- Legend: ⏱ time estimate · 🧪 experiment to record in the Work Log (Appendix E) · ⚠️ watch out · 💡 key concept · 📚 recommended reading · 🔭 Langfuse-specific step.
- A concept is explained where it is first used, in a paragraph that starts with 💡 **How … works** or 💡 **Why …**: read it before doing the step. Section 0.5 lists them all, so that you can find one again later. If a step relies on a concept that no paragraph explains, treat it as a gap in the tutorial and add the paragraph there.
- Every non-trivial decision becomes an ADR (Architecture Decision Record, Appendix C). Every experiment with numbers goes in the Work Log. These two habits are worth more than any tool.
- Nothing is created ahead of time. Every crate, folder, dependency, `just` recipe, configuration value and cloud resource appears in the module that first needs it, together with the reason it exists. If a step ever asks you to prepare something for a later module, treat it as a mistake in the tutorial and move it.

### 0.2 Progress

- [X] Module 1 · Foundations: LLMs, harnesses, agents
- [X] Module 2 · Environment and Rust repository setup
- [X] Module 3 · Enabling Amazon Bedrock
- [X] Module 4 · First Bedrock call from Rust
- [ ] Module 5 · Enabling Langfuse and seeing the calls
- [ ] Module 6 · Systematic prompt engineering with Langfuse Prompt Management
- [ ] Module 7 · Tool calling and the first agent loop
- [ ] Module 8 · Context engineering
- [ ] Module 9 · The guiding project's pipeline
- [ ] Module 10 · Model router
- [ ] Module 11 · Evaluating models with datasets and experiments
- [ ] Module 12 · Regression test suite
- [ ] Module 13 · Security, privacy and guardrails
- [ ] Module 14 · Production observability with Langfuse
- [ ] Module 15 · Deploying to production
- [ ] Module 16 · Optional deep dives

### 0.3 The guiding project

So that you are not learning in a vacuum, every module builds a single application that grows step by step. It is inspired by an internal document, *AI Prompt Processing Pipeline* (Notion), taken as a starting point rather than a specification: that document's section on models is dated and should be ignored.

**A conversational assistant for a healthcare service**, which for every user message runs:

1. `STATE_READ` · reads conversation state and user profile from the database (no LLM call).
2. `SAFETY` · an LLM classifier acting as a gate: if it detects acute risk, it returns an escalation message and stops.
3. `RETRIEVAL` · searches a knowledge base (vector search, no LLM call).
4. `GENERATION` · the main LLM call with **structured output**: the reply text plus signals (affective state, topics, knowledge base references, profile hints).
5. `GUARDRAILS` · a second LLM call that validates the reply; on failure, `GENERATION` re-runs with corrective instructions.
6. `STATE_WRITE` · synchronous conversation state update.
7. `PROFILE_UPDATE` · an **asynchronous** LLM call that updates the user profile after the reply has been delivered.

Almost every topic in this tutorial shows up in that pipeline: different prompts per step, different models per step (the router), context management (a token-budgeted turn window, a compact profile), guardrails, streaming with parallel validation, atomic updates, one Langfuse trace per message with a span per step, tests per step.

⚠️ The healthcare domain is real but this is a **playground**: use synthetic data only, never real patient data.

### 0.4 Starting decisions (explicit assumptions)

| Topic | Decision | Why |
|---|---|---|
| Language | Rust (edition 2024), `tokio` runtime | Explicit requirement. Excellent for understanding the harness by hand: no magic. |
| Provider | Amazon Bedrock through the **Converse API** with the official AWS SDK for Rust | Converse is Bedrock's unified API: the same types for every model, plus tool use, streaming, caching and structured output. |
| Region | An **EU** region (`eu-west-1` or `eu-central-1`) and `eu.*` inference profiles | Data residency (GDPR). See Module 3. |
| Models | The Claude family on Bedrock as the baseline, plus at least one non-Anthropic model (Amazon Nova, Llama, Mistral) so that comparison and routing are meaningful | A router and an evaluation suite only make sense with real alternatives. |
| LLM engineering platform | **Langfuse**, used across all four of its areas: tracing, prompt management, evaluation (datasets, experiments, LLM-as-a-judge, annotations), dashboards and alerts | It is open source and has an EU cloud region; it integrates from Rust over OpenTelemetry and its public API. See Module 5. |
| Agent frameworks | **None** in the core: the harness is written by hand. Rig (the most mature Rust framework) and Temporal (durable execution) are optional comparisons in Module 16 | Writing the loop once is the fastest way to understand what frameworks do for you. |
| Persistence | SQLite locally (`rusqlite` or `sqlx`), PostgreSQL with `pgvector` when vector search is needed | Easy to start, realistic for production. |

### 0.5 Where concepts are explained

Every entry points to the step whose paragraph explains it. The foundations are in Module 1; the rest sit next to the step that first needs them.

| Concept | Step |
|---|---|
| What an LLM is, tokens and the context window | 1.1, 1.2 |
| Message roles and the system prompt | 1.3 |
| Inference parameters: temperature, `max_tokens`, thinking and effort | 1.4 |
| Tool calling, the harness and agents | 1.5 |
| Structured output, the two routes | 1.6 |
| Streaming | 1.7 |
| Workflows versus agents, the composition patterns | 1.8, 1.9, 1.10 |
| Non-determinism | 1.11 |
| Trace, observation, score | 1.12 |
| Cross-region inference profiles | 3.4 |
| Input, output and cache token prices | 3.11 |
| Retries, exponential backoff and jitter | 4.4 |
| Prompt caching | 4.7 |
| Async Rust: futures, the runtime, `Send` and `'static`, cancellation | 4.8 |
| Tracing: spans, events, subscribers | 4.10 |
| OpenTelemetry: context, exporters, flushing | 5.6 |
| Why few-shot examples can hurt | 6.2 |
| Native structured output and constrained decoding | 6.8 |
| Reading a classifier: confusion matrix, precision, recall, noise | 6.10 |
| The Model Context Protocol | 7.9 |
| Embeddings, vector search, chunking, recall@k | 8.7 |
| Hallucination and grounding | 8.7 |
| Idempotency and atomicity | 9.6 |
| Server-Sent Events | 9.7 |
| Self-reported confidence | 10.2 |
| Circuit breakers and token buckets | 10.3 |
| Biases of an LLM judge | 11.4 |
| Repetitions and confidence intervals | 11.9 |
| OIDC federation between CI and AWS | 12.4 |
| Health data under the GDPR | 13.4 |

---

## Module 1 · Foundations: LLMs, harnesses, agents

⏱ 3–4 hours · reading and notes only, no code.

**Goal.** Build a precise vocabulary before writing a line of code.

**Why it matters.** Seventy percent of the mistakes in agentic applications come from a wrong mental model of what an LLM does and does not do: you write a prompt that "works" without knowing why, and then you cannot debug it.

### Steps

- [X] 1.1 💡 **What an LLM is, for the person integrating it.** A function `(token sequence) → (distribution over the next token)`, sampled in a loop. It has no memory between calls: everything it "knows" about the conversation is what you send it every time. Write that sentence in your own words in the Work Log.
- [X] 1.2 💡 **Tokens, context window, cost.** Tokens are the unit of measurement for everything: cost (price per million input and output tokens), limits (the context window), latency (output tokens are paid for in time). Italian text runs at roughly one token per 3–4 characters. The context window of Claude 5 models on Bedrock is 1M tokens, but "it fits" is not "it works well": quality degrades when you fill the window with noise.
- [X] 1.3 💡 **Message roles.** `system` (operator instructions: who you are, what you do, in what format you answer), `user`, `assistant`. The system prompt is the most powerful lever you have.
- [X] 1.4 💡 **Inference parameters: how the model picks tokens, and how much it thinks.** These are the knobs you send with every request. They do not change what the model knows; they change how it produces text. They come in two generations, and which ones a model accepts depends on the model.
  - **Sampling knobs.** At every step the model assigns a probability to each candidate token, and one of them must be picked. After "The sky is", for example: *blue* 60%, *clear* 25%, *cloudy* 10%, everything else 5%. `temperature` sets how much risk to take: low means the most likely token almost always wins (predictable, conservative answers); high flattens the distribution, so *cloudy* comes out more often (more varied answers, and more wrong ones). `top_p` cuts the tail: at 0.9 only the most likely tokens that together reach 90% stay in play, which keeps absurd tokens out. Tune one of the two, not both.
  - **Zero is not deterministic.** Even at `temperature = 0` the probabilities wobble very slightly from one call to the next, because of how GPUs compute your request batched together with other people's. When two tokens are nearly tied, that wobble is enough to swap them, and from that token on the whole text takes a different path.
  - **`max_tokens` is a ceiling, not a target.** The model does not know it and does not try to fill it. If it reaches it, the reply stops mid-sentence with `stop_reason = max_tokens`, and a JSON answer cut that way is unusable. You only pay for the tokens actually generated, so a generous ceiling costs nothing. `stop_sequences` stops generation as soon as a given string appears; with structured output you rarely need it.
  - **Thinking and effort.** Recent Claude models can write internal reasoning before the final answer. You do not see it, but you pay for it as output tokens, and it counts toward `max_tokens`: a low ceiling can run out before the answer arrives. It is *adaptive* because the model decides whether and how much to think, based on how hard the request is. You steer it with **effort**, from `low` (little reasoning, fewer tokens, faster and cheaper) to `max` (more reasoning, slower and more expensive, better on hard problems). The default is `high`.
  - **Which model accepts what.** Claude 5 models (Opus, Sonnet, Fable) reject `temperature` and `top_p` with an error: effort is your only lever. Claude Haiku 4.5 is the other way round: it accepts `temperature` and `top_p`, rejects effort, and thinks only when given a fixed token budget. Non-Anthropic models (Nova, Llama, Mistral) accept the sampling knobs and have their own reasoning settings, if any. So the harness must never send the same parameters to every model (4.6, 10.1).
- [X] 1.5 💡 **Tool calling (function calling).** The model executes nothing: it emits a `tool_use` block with a name and JSON arguments, you run the function and send back a `tool_result` block. The loop `call → tool_use → execute → tool_result → call…` **is the harness**. An "agent" is a harness in which the model decides which tools to use and when to stop.
- [X] 1.6 💡 **Structured output.** Asking the model to answer with JSON that conforms to a schema. There are two ways: force a tool whose schema is the one you want, or use the native structured output feature (on Bedrock: `outputConfig.textFormat` with `json_schema`). You need it everywhere in the guiding project.
- [X] 1.7 💡 **Streaming.** Receiving tokens as they are produced instead of all at the end. It changes perceived latency (time to first token) and complicates the code (partial events, tool use arriving in pieces).
- [X] 1.8 📚 **Workflows versus agents.** Read [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic). Learn the distinction: a **workflow** is where your code decides the flow and the LLM handles individual steps; an **agent** is where the LLM decides the flow. The guiding project is a workflow with agentic parts, which is the right shape for most products.
- [X] 1.9 📚 **The five composition patterns.** Prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer, all described in [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents). For each one write a line in the Work Log: "in the guiding project I would use this for…". (Hint: `SAFETY` is routing, `GUARDRAILS` is evaluator-optimizer.)
- [X] 1.10 💡 **When you do NOT need an agent.** Four questions: is the task multi-step and hard to specify up front? Does the value justify the extra cost and latency? Is the model capable at this kind of task? Are errors recoverable? If any answer is no, stay with a single call or a workflow.
- [X] 1.11 💡 **Non-determinism.** Even at `temperature = 0`, two identical calls can produce different output. This changes everything about testing (Module 12): you do not test "the output is X", you test "the output satisfies properties P, over N repetitions, above a threshold".
- [X] 1.12 💡 **Observing an LLM system.** Three words we will use constantly: **trace** (everything that happens for one request), **observation** (a piece of a trace: a generic span, a *generation* meaning a model call, an event), **score** (a number or a label attached to a trace or an observation: quality, feedback, guardrail outcome). This is the Langfuse data model, and roughly the industry's.
- [X] 1.13 Fill in the glossary (Appendix A) in your own words. If a definition does not come to you, you have not understood the concept yet.

**Done when** you have written definitions in the Work Log for: token, context window, system prompt, tool use, harness, agent, workflow, structured output, streaming, effort, trace, observation, score.

**Self-check.** Explain out loud, in two minutes, why "the LLM remembers the conversation" is false, and what your code actually does to create that illusion.

**Going deeper.** [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) (Anthropic Engineering): we return to both in Modules 7 and 8.

---

## Module 2 · Environment and Rust repository setup

⏱ 2–3 hours.

**Goal.** One Rust crate that builds and runs, a `justfile` with the only two commands that make sense today, and a CI that checks formatting, lints and tests on every pull request.

**Why it matters.** This is the smallest project that already has every habit you will keep: format, lint, test, CI, ADR, commit. The final layout of the project, listed in `AGENTS.md`, is where you arrive, not where you start: each crate appears in the module that needs it, so you always know why it exists.

### Steps

- [X] 2.1 **Toolchain and developer tools.** Rust is pinned in `.tool-versions` (asdf); verify with `rustc --version`. Then install the command-line tools the tutorial relies on. They are programs you run in the terminal, not libraries: they never appear in `Cargo.toml` or in your code, they only speed up your work loop. `cargo install --locked <name>` compiles a tool from source into `~/.cargo/bin`, using the exact dependency versions its authors tested; [`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall) downloads a prebuilt binary instead, which is much faster (`cargo binstall <name>`).
  - **Install now.**
    - [`cargo-nextest`](https://nexte.st/) replaces `cargo test` as the test runner. It runs every test in a separate process, in parallel, so tests cannot interfere with each other; it is up to three times faster, retries flaky tests on request, and selects tests with a filter expression. The tutorial runs every test through it, including the ignored contract tests of Module 12 (`cargo nextest run --run-ignored only`).
    - [`just`](https://github.com/casey/just) is a command runner. You write short named recipes in a `justfile`, such as `check` and `test`, and launch them with `just check`. It borrows the syntax of `make` without being a build system, so it avoids `make`'s quirks. It keeps the project's long commands in one place.
    - [`bacon`](https://dystroy.org/bacon/) runs in a terminal next to your editor and re-checks the code in the background on every save, showing compiler errors, Clippy warnings or failing tests as you work (`c` switches to Clippy, `t` to tests). It is the shortest feedback loop you can get. ⚠️ Many guides suggest `cargo-watch` for this job: its repository is archived and no longer maintained.
  - **Install when the module that needs it arrives.**
    - [`cargo-insta`](https://insta.rs/) is the command-line companion of the `insta` snapshot testing library (Module 12): `cargo insta review` walks you through every changed snapshot and lets you accept or reject it.
    - [`cargo-deny`](https://embarkstudios.github.io/cargo-deny/) checks your dependencies against four rule sets: security advisories, licenses, banned crates and duplicate versions, and allowed sources. It runs in CI and before a release (Module 15).
    - [`cargo-lambda`](https://www.cargo-lambda.info/) builds, runs locally and deploys Rust functions on AWS Lambda (Module 15, and only if you choose Lambda). It needs Zig to link binaries for Linux: the Homebrew installer adds Zig for you, `cargo install` does not, and `cargo lambda system --install-zig` tells you how.
- [X] 2.2 **The first crate.** At the root of the repository run `cargo init`. It creates a single package: a `Cargo.toml` named after the folder, with edition 2024 and an empty `[dependencies]` section, and a `src/main.rs` that prints "Hello, world!". It also appends `/target` to `.gitignore`, because `target/` holds build output that must never be committed. Check that `cargo run` prints the greeting. That is all the project needs today. 💡 A Cargo **workspace**, several crates sharing one `Cargo.lock` and one `target/`, becomes useful the moment you need a second crate with its own boundary: that happens in Module 4, and you convert the project then.
- [X] 2.3 💡 **`Cargo.toml` and `Cargo.lock`.** `Cargo.toml` says what you want: the package's name and edition, and the dependencies with the range of versions you accept. `Cargo.lock`, which appears after the first build, records the exact versions Cargo resolved, so your machine and CI build the same thing: commit it. Dependencies arrive with the code that uses them. When a module needs a library, add it with `cargo add <name>` at that moment, never in advance, and keep the version Cargo picks rather than a number read in a tutorial.
- [X] 2.4 **Formatting, lints and the first test.** `cargo fmt --check` verifies formatting, and `cargo clippy --all-targets -- -D warnings` runs the linter and fails on any warning. `cargo nextest run` fails when it finds no test at all, by design, so write the first one now: move the greeting into a `greeting()` function and add a `#[test]` that checks what it returns. Then create a `justfile` with two recipes, `check` for formatting and lints and `test` for the tests. Later modules add their own recipes when they need them.
- [X] 2.5 **CI on GitHub Actions.** If the repository is not on GitHub yet, create it and push: CI needs it. Add a single `ci.yml` workflow that on every pull request runs the same checks as `just check` and `just test`, **with no credentials**: nothing in this module talks to AWS or Langfuse. The nightly workflow for tests against real services arrives in Module 12, together with those tests.
- [X] 2.6 Write `docs/adr/0001-project-layout.md` from `docs/adr/0000-template.md`: one crate today, a workspace when a second crate is needed, every piece created by the module that uses it. It is your first ADR: five lines is fine.
- [X] 2.7 Commit (English, imperative: `Add the first crate, justfile and CI`).

**Done when** `cargo run` prints the greeting, `just check` and `just test` pass, and CI is green on a pull request.

**Self-check.** What is the difference between `Cargo.toml` and `Cargo.lock`, and why do you commit the lock file? Why does the project not need a workspace yet?

---

## Module 3 · Enabling Amazon Bedrock

⏱ 2–3 hours, much of it waiting on the AWS console.

**Goal.** An AWS account with Bedrock enabled in the EU, least-privilege credentials, models enabled, a budget in place and a successful first call from the CLI.

**Why it matters.** Bedrock's access model is different from an "API key" provider: IAM, regions, model enablement, quotas. Getting this wrong produces cryptic errors (`AccessDeniedException`, `ValidationException` on a model id) that look like bugs in your code.

### Steps

- [X] 3.1 **Account.** Use a dedicated **sandbox** AWS account (if the company has AWS Organizations, ask for one; otherwise a personal account). MFA on the root user, and never work as root.
- [X] 3.2 **Identity.** Prefer IAM Identity Center (SSO) with a permission set; otherwise an IAM user with MFA and rotated access keys. To get going, the managed policy `AmazonBedrockFullAccess` is fine; **before Module 15** replace it with a policy granting only `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (these are the actions Converse and ConverseStream use too) on the ARNs of the inference profiles you actually use.
- [X] 3.3 **AWS CLI.** Already installed (v2.36). Configure a profile: `aws configure sso` or `aws configure --profile bedrock-playground`. Verify with `aws sts get-caller-identity --profile bedrock-playground`.
- [X] 3.4 **Region.** Pick `eu-west-1` (Ireland) or `eu-central-1` (Frankfurt) as your home region. ⚠️ Milan (`eu-south-1`) has a thinner model catalogue. 💡 With **cross-region inference** (an inference profile prefixed `eu.`) Bedrock routes across EU regions while staying inside the geography: more capacity, and data that does not leave the EU. The `global.` prefix can leave the EU: do not use it in this project.
- [X] 3.5 **Enabling models.** Console → Bedrock → *Model catalog*. **Anthropic** models require a *use case* form, filled in once per account, with access granted immediately on submission. Other providers' models (Amazon Nova, Llama, Mistral) need no form, and you enable them in Module 11, when you first compare against them. The AWS Marketplace subscription is created automatically on first invocation if the identity has the right permissions.
- [X] 3.6 **Find the right ids.** Ids change: do not copy them from tutorials, read them from your account.

  ```bash
  aws bedrock list-foundation-models --region eu-west-1 --profile bedrock-playground --by-provider anthropic --query 'modelSummaries[].modelId'
  aws bedrock list-inference-profiles --region eu-west-1 --profile bedrock-playground --query 'inferenceProfileSummaries[].inferenceProfileId'
  ```

  Example shapes, to be verified: `eu.anthropic.claude-sonnet-5`, `eu.anthropic.claude-opus-5`, `eu.anthropic.claude-haiku-4-5-20251001-v1:0`. Now the project needs its first configuration values: copy `.env.example` to `.env`, which git ignores, and fill in the AWS section with your profile, the region, a **fast and cheap** model (`BEDROCK_MODEL_FAST`, typically Haiku) and a **main** one (`BEDROCK_MODEL_MAIN`, typically Sonnet).
- [X] 3.7 **Quotas.** Console → Service Quotas → Bedrock: look for *requests per minute* and *tokens per minute* for your chosen models. Write them down: they are the ceiling the router (Module 10) has to respect. Request an increase only if you need one.
- [X] 3.8 **Budget.** AWS Budgets: a monthly budget (say 30 €) with email alerts at 50% and 80%.
- [X] 3.9 **Invocation logging.** Console → Bedrock → *Settings* → *Model invocation logging* to CloudWatch Logs. Useful in the first days of debugging; once you have Langfuse (Module 5) it becomes redundant. ⚠️ Those logs contain full prompts: in production with personal data this choice must be revisited (Module 13).
- [X] 3.10 🧪 **First call from the CLI.** Load `.env` into your shell first, so that `$BEDROCK_MODEL_FAST` is defined: `set -a; source .env; set +a`.

  ```bash
  aws bedrock-runtime converse \
    --region eu-west-1 --profile bedrock-playground \
    --model-id "$BEDROCK_MODEL_FAST" \
    --messages '[{"role":"user","content":[{"text":"Answer with a single word: ready?"}]}]' \
    --inference-config '{"maxTokens":50}'
  ```

  Look at the response: `output.message.content`, `stopReason`, `usage.inputTokens`, `usage.outputTokens`, `metrics.latencyMs`. Record the tokens and the latency in the Work Log: that is your first measurement.
- [X] 3.11 📚 **Pricing.** Read the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) for your chosen models. Work out by hand what the call you just made cost. Learn the difference between input tokens, output tokens, **cache writes** and **cache reads** (cache reads cost roughly 10% of normal input). You will enter these prices into Langfuse in Module 5.
- [X] 3.12 💡 **A note on "Claude Platform on AWS".** Since 2026 Anthropic also offers access operated directly by Anthropic inside AWS (SigV4 authentication, Marketplace billing) with feature parity with the Anthropic API and unprefixed model ids. It is an alternative to Bedrock, not the same thing: Bedrock is operated by AWS and has its own multi-vendor catalogue, Guardrails, Knowledge Bases and AgentCore. This tutorial stays on Bedrock because the point is to learn the provider; we come back to it in Module 16.

**Done when** the call in 3.10 answers, the budget is active, and `.env` holds two valid EU-region model ids.

**Self-check.** What happens if you call a model with a `us.*` id from `eu-west-1`? And why is an `eu.*` inference profile preferable to a bare, unprefixed model id?

---

## Module 4 · First Bedrock call from Rust

⏱ 4–6 hours.

**Goal.** The project becomes a workspace with its first two libraries: `llm-core`, which defines the `LlmClient` trait, and `llm-bedrock`, which implements it with a simple call, streaming, error handling, and token and latency measurement.

**Why it matters.** This is where the most important boundary in the architecture is born: the harness will only ever talk to `LlmClient`, never to the SDK.

### Steps

- [X] 4.1 **From one crate to a workspace.** You now need two libraries with a boundary between them: `llm-core` holds neutral types and must not know about AWS, `llm-bedrock` implements them over the AWS SDK. Turn the root `Cargo.toml` into a *virtual manifest*, a workspace with no package of its own: a `[workspace]` table with `members = ["crates/*"]` and `resolver = "3"` (without it Cargo warns and falls back to the old dependency resolver). Move the binary into `crates/app` with `cargo new crates/app`, carry your `main.rs` over and delete the root `src/`; then create the libraries with `cargo new --lib crates/llm-core` and `cargo new --lib crates/llm-bedrock`. 💡 When a dependency is used by more than one crate, declare its version once under `[workspace.dependencies]` and write `name = { workspace = true }` in each crate.

  Then define the neutral types in `llm-core`, only the ones this module uses; do not copy the SDK's types. `llm-core` needs `thiserror` (typed errors), `serde_json` (the free-form `extra` field), `futures` (the `Stream` type used for streaming) and `async-trait` (async methods in a trait you can use as `dyn LlmClient`).

  ```rust
  pub enum Role { User, Assistant }
  pub enum ContentBlock { Text(String), CachePoint }   // ToolUse and ToolResult arrive in Module 7
  pub struct Message { pub role: Role, pub content: Vec<ContentBlock> }
  pub struct LlmRequest { pub model: ModelId, pub system: Vec<SystemBlock>, pub messages: Vec<Message>,
                          pub max_tokens: u32,
                          pub extra: serde_json::Value /* model-specific fields, e.g. effort */ }
                          // output_schema arrives in Module 6, tools in Module 7
  pub struct LlmResponse { pub message: Message, pub stop_reason: StopReason, pub usage: Usage, pub latency: Duration }
  pub struct Usage { pub input_tokens: u32, pub output_tokens: u32, pub cache_read_tokens: u32, pub cache_write_tokens: u32 }

  #[async_trait::async_trait]
  pub trait LlmClient: Send + Sync {
      async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError>;
      async fn stream(&self, req: LlmRequest) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError>;
  }
  ```

- [X] 4.2 Make `llm-bedrock` depend on `llm-core` (a path dependency inside the workspace) and add `aws-config` (feature `behavior-version-latest`), which reads credentials and region from your AWS profile, and `aws-sdk-bedrockruntime`, the Bedrock Runtime client. Build the client once, because it is expensive, and share it with `Arc`.

  ```rust
  let cfg = aws_config::defaults(BehaviorVersion::latest()).region(Region::new("eu-west-1")).load().await;
  let client = aws_sdk_bedrockruntime::Client::new(&cfg);
  ```

- [X] 4.3 Implement `complete` with `client.converse()`: `model_id`, `system(SystemContentBlock::Text(..))`, `messages(..)`, `inference_config(InferenceConfiguration::builder().max_tokens(..))`. Map the response: `output()` → `ConverseOutput::Message`, `stop_reason()`, `usage()`, `metrics().latency_ms()`. 📚 Check [`aws_sdk_bedrockruntime` on docs.rs](https://docs.rs/aws-sdk-bedrockruntime) for the exact names: the SDK's builders change between versions.
- [X] 4.4 **Errors.** Map the service exceptions to an `LlmError` whose variants matter to the caller: `Throttled` (retryable), `ModelNotReady`/`ServiceUnavailable` (retryable), `ValidationError` (your bug, do not retry), `AccessDenied` (configuration), `ContextTooLong`, `Other`. 💡 The SDK already retries with exponential backoff and jitter: configure it (`RetryConfig`) instead of rewriting it; the router (Module 10) will only add cross-model fallback.

  💡 **How retries work.** A retry helps only with failures that may not happen again: throttling (the quota frees up within the minute), a model still loading, a transient server error. Retrying a validation error returns the same error. *Exponential backoff* waits longer after each failure, for example 1 s, 2 s, 4 s up to a ceiling, so that a service under pressure gets room to recover instead of more load. *Jitter* picks a random wait up to that value: without it, every client that failed at the same moment retries at the same moment and they fail together again. A maximum number of attempts bounds the extra latency, since the caller sees the error only after the last wait. Retrying also costs something: every attempt that reaches the model is paid, and repeating a call that changed something (a write, a booking) must be safe, which is what idempotency means (9.6).
- [X] 4.5 **Streaming.** Implement `stream` with `converse_stream()`: you receive `ConverseStreamOutput` with `MessageStart`, `ContentBlockStart`, `ContentBlockDelta` (pieces of text), `ContentBlockStop`, `MessageStop` and `Metadata` (usage) events. Translate them into your own `LlmEvent`.
- [X] 4.6 **Model-specific parameters.** Converse splits parameters in two. `inferenceConfig` holds the fields every model shares (`maxTokens`, `temperature`, `topP`, `stopSequences`); anything vendor-specific goes in `additional_model_request_fields` (a `Document`). For Claude, `thinking` and `output_config.effort` travel there, among others. Wire `LlmRequest.extra` into that field. ⚠️ Models do not accept the same parameters (see 1.4): Claude 5 rejects `temperature`, Claude Haiku 4.5 rejects `effort`. For now `extra` carries only what you set by hand for a model you chose; the first time a parameter such as effort comes from configuration is Module 6, and that is where you add the safety net (6.4).
- [X] 4.7 **Prompt caching.** Converse supports a `cachePoint` block: it marks the end of the stable prefix (system prompt, tool definitions). Add `ContentBlock::CachePoint` and verify that `usage.cache_read_tokens` becomes greater than zero on the second identical call. If it stays at zero, something in the prefix changes on every call (a timestamp, field ordering).

  💡 **How prompt caching works.** What is cached is not the answer but the work of reading the prompt. A model handles a request in two phases: it first reads every input token and builds an internal representation of each one, which is the expensive part and grows with the length of the prompt, then it generates the output one token at a time. When a request starts with exactly the same tokens as an earlier one, the representation of that prefix is identical, so Bedrock can keep it and compute only what follows; the `cachePoint` tells it where the reusable prefix ends. Four consequences follow.
  - **Only an exact match from the first token counts.** A change at position 50 invalidates everything after it, however identical: a timestamp at the top of the system prompt defeats the cache entirely.
  - **The order of the prompt becomes a cost decision.** What never changes goes first (instructions, schema, rules), then the `cachePoint`, then what changes (profile, retrieved context, the message). Steps 6.7 and 8.9 build on this.
  - **The answer does not change.** The model does the same computation without repeating the part already done. What changes is the cost (a read costs a tenth of the input price, the first write 1.25 times) and, with long prompts, the time to first token.
  - **The cache has a lifetime and limits.** It expires after about five minutes without use, and every read renews it; a one-hour option costs twice the input price to write. A prefix shorter than a minimum, which depends on the model, is silently not cached. The cache belongs to one model (10.5).

  It matters in an agentic application because every call resends everything (1.1): an agent loop with ten tool calls sends the system prompt, the tool definitions and the growing history ten times, and without the cache pays for them ten times at full price. Do not confuse it with caching a response, which Bedrock does not do, nor with the prompt cache of Module 6, which keeps the prompt text downloaded from Langfuse in memory and has nothing to do with the model.
- [X] 4.8 🧪 In `crates/app`, an `app hello` command that makes a call and prints the reply, stop reason, tokens, latency and estimated cost (a hardcoded price table for now). The binary needs `tokio` (the async runtime, started by `#[tokio::main]`), `anyhow` (simple error handling for a binary), `dotenvy` (loads `.env`) and `clap` (parses the command line); add a `run` recipe to the `justfile`. Repeat the same call five times: record the latency variance and whether the cache is being read.

  💡 **How async Rust works.** An `async fn` does not run when you call it: it returns a *future*, a value describing work that can pause while it waits (for the network, a timer, a file) and resume later. Nothing happens until someone *polls* the future, and that someone is a runtime: `tokio` keeps a pool of threads and drives thousands of futures on them, running whichever is ready while the others wait. `#[tokio::main]` starts the runtime and runs `main` on it; `.await` is where a future may pause and give the thread back. Three consequences run through this project.
  - **Concurrency happens only when you ask for it.** Two `.await`s in a row run one after the other. `tokio::join!` waits for several futures at once, and `tokio::spawn` hands a future to the runtime to run on its own (both in 9.2).
  - **What a spawned future holds must be `Send` and `'static`.** A spawned future may move between threads and outlive the function that created it, so its data must be safe to move between threads (`Send`) and owned rather than borrowed (`'static`). This is why `LlmClient::stream` returns a `BoxStream<'static, …>`, and why a trait with async methods used as `dyn LlmClient` needs `async-trait`.
  - **Cancelling is dropping.** A future that nobody polls any more simply stops at its last `.await`, with no exception. Timeouts become easy (`tokio::time::timeout` drops the future when time runs out), but whatever the future had half done stays half done, which is why writes must be atomic or idempotent (9.6).

  Never block a runtime thread with slow synchronous work, such as a long computation or `std::thread::sleep`: every other future on that thread stops too.
- [X] 4.9 🧪 Streaming from the CLI: print tokens as they arrive. Measure **time to first token** and total time. Compare them with the non-streaming call.
- [X] 4.10 Add `tracing`, which records spans and events, and `tracing-subscriber`, which prints them: one span per call with `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `latency_ms` and `stop_reason` fields. In Module 5 these spans become Langfuse *generations* without touching `llm-bedrock`.

  💡 **How tracing works.** `tracing` separates recording from printing. The code records *spans*, intervals of time with a name and fields (`llm_call` with `model`, `input_tokens`…), and *events*, points in time inside a span (a warning, a retry). A span opened inside another becomes its child, so one request produces a tree, and that tree is what Langfuse shows as a trace. Where the records go is decided elsewhere, by a *subscriber* made of *layers*: `tracing-subscriber` prints them on the terminal, and in Module 5 a second layer turns the same spans into OpenTelemetry spans for Langfuse without changing a line of `llm-bedrock`. That separation is the reason to instrument with `tracing` rather than calling Langfuse directly. ⚠️ In async code a span must be attached to the future (`.instrument(span)` or `#[instrument]`), never held open across an `.await` with `span.enter()`: while the future is paused, other work running on the same thread would be recorded inside it.

**Done when** `app hello` and `app hello --stream` work, throttling and validation errors are distinguishable in the logs, and `cache_read_tokens > 0` on the second call.

**Self-check.** Why must `llm-core` not depend on `aws-sdk-*`? (So you can test the harness without the provider, add a second provider without touching the harness, and keep the harness independent of the AWS SDK.) Which errors are retryable and which are not?

**Going deeper.** The official [Bedrock Runtime examples for Rust](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_bedrock-runtime_code_examples.html) in the *AWS SDK for Rust Developer Guide*; the [`Converse`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) and [`ConverseStream`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html) API reference.

---

## Module 5 · Enabling Langfuse and seeing the calls

⏱ 4–6 hours.

**Goal.** A live Langfuse project in the EU cloud region, the Rust app exporting every call as a *generation* with tokens, cost, model and latency, and you being able to read a trace.

**Why it matters.** From here on, every experiment in this tutorial is read in Langfuse, not in the logs. Seeing the calls while you learn shortens the feedback loop more than any amount of reading. And it is the tool you want to master.

### Steps

- [X] 5.1 📚 🔭 **The data model.** Read [Observability Data Model](https://langfuse.com/docs/observability/data-model) in the Langfuse docs. Fix these in your mind: **trace** (one end-to-end request, with `input`, `output`, `user_id`, `session_id`, `tags`, `metadata`, `release`, `version`, `environment`), **observation** (span, generation, event, plus the agentic types agent/tool/chain/retriever/evaluator; nestable), **generation** (one model call: model, parameters, usage, cost, linked prompt), **score** (numeric, categorical or boolean; on a trace, an observation or a session), **session** (several traces from the same conversation), **dataset** and **dataset run** (you meet these in Module 11).
- [ ] 5.2 🔭 **Where Langfuse runs.** Langfuse Cloud, EU region (`https://cloud.langfuse.com`, data in Ireland `eu-west-1`, the same geography as your Bedrock). Zero infrastructure, and the free tier is enough for this tutorial. Self-hosting is out of scope for this project: running Langfuse teaches its architecture, not agentic applications. Check the free plan's limits and write the choice in an ADR with "data residency" as the rationale.
- [ ] 5.3 🔭 Create an organization and a **project** (`agentic-playground`). Generate an **API key** pair (public and secret) and put them in `.env`. Langfuse keys are per project: different environments (local, dev, prod) can be different projects **or** the same project with an `environment` attribute on traces. For this tutorial use one project plus `environment`.

  ⚠️ Sign up on `https://cloud.langfuse.com`, the EU region: the US region is `us.cloud.langfuse.com`, and an organization stays in the region where you create it.

  Verify the keys with the official [Langfuse CLI](https://github.com/langfuse/langfuse-cli), a command-line tool like those of 2.1 but distributed through npm, so it needs Node.js: `npm install -g @langfuse/cli`. `langfuse --env .env api projects list` reads the keys from `.env` and must return `agentic-playground`. The CLI covers the whole public REST API (`langfuse api help` lists the resources) and its default host is the EU region: later it lets you inspect prompts, datasets and scores without writing code.
- [ ] 5.4 🔭 **Models and prices.** Langfuse computes cost from `usage` only if it knows the model. Under *Settings → Models*, add definitions for your Bedrock model ids (matched by a regex on the name, for example `(?i)^eu\.anthropic\.claude-haiku-4-5.*`) with per-token prices for input, output, cache read and cache write, taken from 3.11. Without this step you will see tokens but no cost. ⚠️ Langfuse already ships managed definitions whose regex accepts the `eu.` prefix, but they carry Anthropic API prices, 10% below EU Bedrock: create your own, which take priority over managed ones (`langfuse api models create`, or the UI). A price applies only when its key matches a key of the generation's usage details exactly, so use the keys you will send in 5.7: `input`, `output`, `cache_read_input_tokens`, `cache_creation_input_tokens`.
- [ ] 5.5 💡 **How you reach Langfuse from Rust.** Langfuse has no official Rust SDK; it has two doors open to any language: the **OpenTelemetry endpoint** (`/api/public/otel`, OTLP over HTTP, Basic authentication with `public:secret`) and the **public REST API**. Use OTel for tracing and the API for prompts, datasets, scores and experiments. Useful crates: [`opentelemetry-langfuse`](https://github.com/genai-rs/opentelemetry-langfuse) (a builder for an OTLP exporter preconfigured for Langfuse; ⚠️ it can lag behind the OpenTelemetry releases, and every OpenTelemetry crate in the chain must share one version of `opentelemetry`, so check its dependencies before choosing it) and [`langfuse-ergonomic`](https://github.com/genai-rs/langfuse-ergonomic) (a public-API client with builders, on top of the OpenAPI-generated [`langfuse-client-base`](https://github.com/genai-rs/langfuse-client-base)).
- [ ] 5.6 Create the `observability` crate (`cargo new --lib crates/observability`) and initialize the stack: `tracing` → `tracing-opentelemetry`, which turns tracing spans into OpenTelemetry spans → `opentelemetry_sdk`, which batches and exports them, with an OTLP exporter pointing at Langfuse: `opentelemetry-otlp` over HTTP with protobuf (Langfuse does not accept gRPC), with the `reqwest-blocking-client` and `reqwest-rustls` features, or HTTPS will not work. Give the endpoint in full, `{LANGFUSE_HOST}/api/public/otel/v1/traces`: an endpoint set in code is used as it is, without `/v1/traces` appended. Send two headers: `Authorization: Basic base64(public:secret)` and `x-langfuse-ingestion-version: 4`, without which spans can take up to ten minutes to appear. If `opentelemetry-langfuse` supports your `opentelemetry` version, its builder does the same. Keep the `fmt` layer for development, on standard error so that the reply on standard output stays clean, and give the OpenTelemetry layer its own filter (a `Targets` with your crates) so that the AWS SDK's spans are printed but not exported. Use the batch exporter, and flush explicitly on shutdown, or CLI binaries will lose their last traces.

  💡 **How OpenTelemetry works.** OpenTelemetry (OTel) is a vendor-neutral standard for traces, metrics and logs: a data model, an SDK per language and a wire protocol, OTLP. A trace is a set of spans that share a trace id; each span carries its own id, its parent's id, start and end times, and attributes, key-value pairs such as `gen_ai.request.model`. The parent-child link travels as *context*: inside a process the library tracks the current span, and between services it travels in an HTTP header (`traceparent`), which is how one trace can cover several services. Spans are not sent one at a time: a batch processor collects them in memory and an exporter ships them in batches over OTLP to a backend, here Langfuse's `/api/public/otel`. This keeps requests fast, and it is why a short-lived CLI loses its last spans unless it flushes the exporter before exiting. Because the protocol is standard, the backend is interchangeable, and the same spans can reach both Langfuse and Datadog through a Collector (14.10). Langfuse decides whether a span is a trace, a generation or a tool call from its attributes (5.7).
- [ ] 5.7 🔭 **Mapping spans onto the data model.** Langfuse reads both the OpenTelemetry GenAI conventions and its own `langfuse.*` attributes. The minimum rules:
  - In Langfuse v4 a trace is no longer a record of its own: it is the set of observations that share a trace id. The **root** span is the whole request, and it carries the overall input and output in `langfuse.observation.input` and `.output`; `langfuse.trace.input` and `.output` are deprecated.
  - The trace-wide context goes on **every** span you want to filter or aggregate, not only on the root, because Langfuse v4 queries observations one by one: `langfuse.trace.name`, `langfuse.session.id`, `langfuse.user.id` (already pseudonymized), `langfuse.trace.tags`, `langfuse.environment`, `langfuse.release` (the app version).
  - The span of an LLM call becomes a generation, with `langfuse.observation.type = "generation"`, `gen_ai.request.model` (or `langfuse.observation.model.name`), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and the cache details in `langfuse.observation.usage_details` (a JSON object, serialized as a string, with `input`, `output`, `cache_read_input_tokens`, `cache_creation_input_tokens`: the keys priced in 5.4); `langfuse.observation.input` and `.output` carry the content (see 5.10 for privacy). For a stream, `langfuse.observation.completion_start_time` (ISO 8601) records when the first token arrived, the time measured in 4.9.
  - Other spans get the most specific type: `"agent"` for the agent loop, `"tool"` for a tool call, `"retriever"` for a search, `"guardrail"` for `SAFETY` and `GUARDRAILS`, `"chain"` or `"span"` for any other pipeline step (Modules 7 to 9).
  - Export each span once, complete: Langfuse v4 does not deduplicate a span id sent twice. `tracing-opentelemetry` already exports a span when it closes.
  📚 Check the exact names in the [OpenTelemetry page](https://langfuse.com/integrations/native/opentelemetry) of the Langfuse docs, and the v4 rules in [Migrate custom ingestion to Langfuse v4](https://langfuse.com/integrations/native/opentelemetry/migration-to-v4): the attribute list evolves.
- [ ] 5.8 **Observability as a decorator.** Observability is a cross-cutting concern, so keep it out of the provider code. In `observability`, write `TracedClient<C: LlmClient>`, which implements `LlmClient` itself: it opens the span of the call, calls the client inside, and sets the 5.7 attributes from your neutral types (`LlmRequest`, `LlmResponse`, `LlmEvent`, `LlmError`), so it works for any provider. Move the span from 4.10 into it, so that `llm-bedrock` no longer needs `tracing`, and build the client as `TracedClient::new(BedrockClient::new(region).await)`. For a stream, collect the reply as it passes through and record the generation when the stream ends. Attributes known only after the span is created go through `OpenTelemetrySpanExt::set_attribute`, because `tracing` fields must be declared when the span opens. The attributes shared by every span of a trace (environment, release, trace name) belong in an OpenTelemetry span processor, which adds them as each span starts, so the code that opens spans does nothing for them. Write the choice in an ADR.
- [ ] 5.9 🧪 Run `app hello` three times. In Langfuse open *Tracing → Traces*: you should see three traces, each with one generation, showing model, tokens, **cost** and latency. Open a generation and read its input, output and usage. If cost is missing, go back to 5.4. If the trace is missing, check the flush and the credentials.
- [ ] 5.10 💡 🔭 **Content and privacy, first rule.** Today the traces carry full prompts and replies, and that is fine: the data is synthetic and the project is only yours. Real personal data would change that, and Module 13 is where you add redaction and sampling, when there is something to protect. For now read the two tools Langfuse documents, [masking](https://langfuse.com/docs/observability/features/masking) and [data retention](https://langfuse.com/docs/administration/data-retention), and write in the Work Log what you would protect first. ⚠️ Masking does not happen at ingestion: with plain OpenTelemetry it happens either in the application, before the OTLP export, or in an OpenTelemetry Collector, after the data has left the application (ready-made masking hooks exist only in the Python and JS/TS SDKs). Project retention, which deletes traces older than a number of days, is available only on the Pro and Enterprise plans; the free plan has only its 30-day access window.
- [ ] 5.11 🧪 Simulate a conversation from the CLI (`app chat` with three messages): each message is a trace, all sharing the same `session.id`. Open *Sessions* in Langfuse: you should see the whole conversation in order. Then *Users*: you should see the synthetic user with cumulative cost.
- [ ] 5.12 🧪 Add your first **score** through the API, adding `langfuse-ergonomic` to `observability` (or calling `POST /api/public/scores` directly): after `app hello`, attach a boolean `smoke_ok = true` score to the trace. This teaches you the mechanism you will use for guardrails (Module 9), user feedback and evaluation (Modules 11 and 14).
- [ ] 5.13 🔭 Take a full tour of the interface and note in the Work Log what each section is for: Tracing (Traces, Sessions, Users, Observations), Prompts, Evaluation (Datasets, Evaluators, Scores, Annotation Queues), Dashboards, Settings (Models, API keys, Members, Retention). It is the map of the next modules.

**Done when** every call from the app appears in Langfuse as a generation with a cost, conversations are grouped into sessions, a score arrives through the API, and your notes on what to protect first are in the Work Log.

**Self-check.** Why do trace attributes (`session.id`, `user.id`, `tags`) belong on the root span and not only on a generation? What is the difference between one Langfuse project per environment and a single project with an `environment` attribute?

**Going deeper.** The [OpenTelemetry](https://langfuse.com/integrations/native/opentelemetry) and [Observability Data Model](https://langfuse.com/docs/observability/data-model) pages in the Langfuse docs; the READMEs of the [`opentelemetry-langfuse`](https://github.com/genai-rs/opentelemetry-langfuse) and [`langfuse-ergonomic`](https://github.com/genai-rs/langfuse-ergonomic) crates.

---

## Module 6 · Systematic prompt engineering with Langfuse Prompt Management

⏱ 8–10 hours, spread over several days.

**Goal.** A repeatable method for writing, measuring and versioning prompts, with Langfuse as the prompt registry and repository files as the fallback, applied to the guiding project's `SAFETY` prompt.

**Why it matters.** "The best possible prompt" does not exist in the abstract: what exists is the best trade-off between quality, cost and latency **for one task, one model and one measured dataset**. Without measurement you go in circles. And without a versioned registry you never know which prompt produced which answer.

### Steps

- [ ] 6.1 📚 Read the [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) in the Claude documentation, then [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), and the prompting notes for the Claude 5 models you will use ([Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5), [Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5)): prompts written for older models tend to be over-prescriptive and degrade output on recent ones.
- [ ] 6.2 💡 **Anatomy of a system prompt.** In order: (1) role and operating context, (2) the task's goal, (3) rules and constraints (positive: "do X" works better than "do not do Y"), (4) output format, (5) examples (few-shot) only if they earn their place, (6) variable data **last**. XML tags (`<context>`, `<rules>`, `<examples>`) help the model separate the parts and are the standard way to delimit user input from everything else.

  💡 **Why examples can hurt.** A model imitates examples more faithfully than it follows descriptions. That is their strength and their risk: it copies their surface (length, wording, language, the order of fields) and, in a classifier, their label distribution. Five examples of which four are `SAFE` push ambiguous messages towards `SAFE`; examples that are all short and explicit teach nothing about the long, indirect message, which is the one that matters most. Examples also cost tokens on every call and anchor the model on their particular cases. So start without examples, add them only when a measured error calls for them, keep them varied and balanced across labels, prefer hard cases to obvious ones, and measure each addition like any other change (6.11).
- [ ] 6.3 📚 🔭 **Prompt Management in Langfuse.** Read [Prompt Management Concepts](https://langfuse.com/docs/prompt-management/data-model). Fix these: a prompt has a **name**, immutable **versions** (1, 2, 3…), **labels** pointing at a version (`production` is what is served by default; `staging`, `latest` and custom labels), a **text** or **chat** type, **variables** with `{{name}}` syntax, a `config` (free-form JSON: put the model, effort and `max_tokens` there), tags, and the ability to compose prompts inside prompts. `production` labels can be **protected** so only certain roles move them.
- [ ] 6.4 🔭 Create the `safety-classifier` prompt in Langfuse (chat type: a system message with the rules and a user message with `{{message}}`), with `config = { "model_role": "fast", "effort": "low", "max_tokens": 1024 }`. Assign the `production` label to v1. The `config` holds hints, not guarantees, and this is the first time a parameter such as effort comes from configuration instead of from code you wrote for a specific model. Map `model_role` to the model ids in `.env` (`fast` to `BEDROCK_MODEL_FAST`, `main` to `BEDROCK_MODEL_MAIN`); if the fast model is Claude Haiku 4.5, which rejects effort, the request fails. Add the safety net now: a small capability table in `llm-bedrock`, keyed by model family, that drops any parameter the target model does not accept (see 1.4 and 4.6), logs a warning, and records the parameters actually sent on the call's span, so that an experiment never believes it changed a variable that was dropped. In Module 10 the table moves into the `ModelRegistry` and the role mapping into the router. The ceiling is generous on purpose: the answer is a small JSON, but a model that thinks before answering spends part of the ceiling on reasoning, and unused ceiling costs nothing.
- [ ] 6.5 🔭 Create the `prompts` crate (`cargo new --lib crates/prompts`) and implement `PromptStore::get(name, label)`:
  1. call `GET /api/public/v2/prompts/{name}?label=production` (through `langfuse-ergonomic`),
  2. **cache in memory** with a TTL (60 seconds is the official SDKs' default, see [Caching](https://langfuse.com/docs/prompt-management/features/caching); 5–10 minutes is fine in production),
  3. **fall back** to the `prompts/<name>.md` file in the repository when Langfuse does not answer (this is the [guaranteed availability](https://langfuse.com/docs/prompt-management/features/guaranteed-availability) pattern from the docs),
  4. return an object with `name`, `version`, `template`, `config` and a `render(vars)` method.
  Add a `just prompts-pull` recipe that downloads snapshots of the `production` versions into a new `prompts/` folder, so every prompt change also shows up in a pull request diff and the fallback stays current.
- [ ] 6.6 🔭 **Link prompts and generations.** When the harness makes a call using a prompt, set `langfuse.observation.prompt.name` and `langfuse.observation.prompt.version` on the generation's span. Opening the prompt in Langfuse then shows every generation that used it, with average cost and latency **per version**: this is the basis of every comparison.
- [ ] 6.7 💡 **Prefix stability.** What does not change between calls (instructions, schema, examples) goes first; what does change (user profile, retrieved context, the message) goes after the `cachePoint`. Every byte that changes in the prefix invalidates the cache. With Langfuse variables this translates to: variables only in the final part of the template.
- [ ] 6.8 💡 **Structured output.** Add an `output_schema` field to `LlmRequest`. Define the output schema as a Rust struct with `serde` (serialization, `derive` feature) and `schemars` (generates a JSON Schema from a struct), and pass the schema in `outputConfig.textFormat` (Converse, `json_schema` type). Alternatively, force a tool with that schema. ⚠️ Not every model supports both routes, and Claude Fable 5.1 does not accept forced tool choice: test this per model and record the result in the `ModelRegistry` (Module 10). The schema lives in the code, not in the prompt: the Langfuse prompt only references the schema by name.

  💡 **How native structured output works.** Asking for JSON in the prompt is a request: the model usually complies, but nothing stops it from adding a sentence before the brace, dropping a required field or inventing one. With a schema passed to the API, the provider constrains generation itself: at every step, the tokens that would make the output invalid against the schema are excluded before one is picked (constrained decoding), so the result parses and has the declared fields and types. Forcing a tool whose input schema is your output schema reaches the same goal through tool use. Three limits remain. Only a subset of JSON Schema is supported, so check which keywords your model accepts. A reply cut at `max_tokens` is still incomplete, so check `stop_reason` before parsing. And validity is not correctness: a well-formed `{"label": "SAFE"}` can still be the wrong label, which only evaluation catches (Module 11).
- [ ] 6.9 💡 **Thinking and effort.** On models that support it (Claude 5, see 1.4) reasoning is adaptive and you tune its depth with `effort` (`low`, `medium`, `high`, `xhigh`, `max`). The main generation runs on a Claude 5 model: try `medium` and `high` there and measure. For a classification task such as `SAFETY`, `low` is usually enough on a Claude 5 model; on Claude Haiku 4.5 the lever does not exist, and what you tune is the prompt itself and, at most, `temperature`. 💡 Before moving to a bigger model, try the current model at higher effort when it supports effort: it is often cheaper. These values belong in the prompt's `config`, so they are versioned together with the text.
- [ ] 6.10 🧪 **Lab: the `SAFETY` prompt in Italian.** Build `datasets/safety/v1.jsonl` with **30–50 synthetic messages** labelled by hand (`SAFE`, `PSYCH_CRISIS`, `MEDICAL_EMERGENCY`), including the ambiguous cases typical of Italian ("non ce la faccio più", "sono stanca di tutto", "mi scoppia la testa"). Measure accuracy, false negatives (the worst case here), latency and cost on `BEDROCK_MODEL_FAST`. For now the runner is a simple script; in Module 11 it becomes a Langfuse experiment.

  💡 **How to read a classifier's results.** Accuracy, the share of correct answers, hides what matters when classes are unbalanced: if 90% of a dataset is `SAFE`, a classifier that always answers `SAFE` scores 90% and misses every crisis. Look instead at the *confusion matrix*, a table of expected label against predicted label, and at two numbers for each risk class.
  - **Recall** is the share of real crises the classifier catches. Its complement is the *false negative* rate, the risk that went unseen: the worst error in this domain.
  - **Precision** is the share of alarms that were real. Its complement, the *false positives*, costs an unnecessary escalation.

  The two trade against each other, and for `SAFETY` you accept lower precision to keep recall high. Numbers from a small dataset are also noisy: over n cases, a rate near 50% is only known to within about ±1/√n at 95% confidence (±18 points with 30 cases, ±10 with 100), a little less near 0% or 100%. A difference between two prompts smaller than that is not evidence of anything. Module 11 turns this into proper confidence intervals (11.9).
- [ ] 6.11 🧪 🔭 Iterate **in Langfuse**: v2 with examples, v3 with more explicit rules about context, v4 with a different `effort` in the `config` if the fast model supports it, otherwise a different `temperature`. One variable at a time. Use the Langfuse **Playground** to try a variant on three or four cases quickly before running the whole dataset (connect the playground to Bedrock under *LLM connections* if it is available for your model; otherwise run it from your own runner). Record every run with the card in Appendix D. Move the `production` label only onto the winning version.
- [ ] 6.12 🧪 Repeat 6.10 with `BEDROCK_MODEL_MAIN`. Does the big model beat the small one? By how much, at what cost and latency? This is your first ADR-worthy decision: "for `SAFETY` we use X because…".
- [ ] 6.13 💡 **Anti-patterns to recognize.** Enormous prompts that "cover every case"; rules in capitals and threats; asking for JSON without a schema; putting user data in the middle of the instructions; changing prompt and model together; judging a prompt on three examples; moving `production` without a dataset to justify it.
- [ ] 6.14 💡 **Prompt injection, first notion.** Anything coming from a user or a retrieved document is **data**, not instructions. Delimit it, and state in the system prompt that content inside certain tags is text to be analyzed. More in Module 13.

**Done when** the `SAFETY` prompt lives in Langfuse with at least three versions measured on the same dataset, the app loads it with caching and a fallback, every generation in Langfuse shows the prompt name and version, and an ADR records which version and which model you use, with the numbers.

**Self-check.** What happens to the app if Langfuse is unreachable for an hour? (It must keep working from cached prompts or from the files: if not, the fallback is not implemented.) Why can a five-example few-shot make a classifier worse?

---

## Module 7 · Tool calling and the first agent loop

⏱ 6–8 hours.

**Goal.** A generic harness in `harness`: a tool registry, an execution loop, safety limits, tests with `FakeLlmClient`, and nested traces in Langfuse. Plus a connection to an MCP server.

**Why it matters.** This is the piece a framework usually hides. Writing it by hand once shows you what a framework gives you, and what it takes away.

### Steps

- [ ] 7.1 💡 Create the `harness` crate (`cargo new --lib crates/harness`) and define `Tool` as a trait: `name()`, `description()`, `input_schema()` (generated by `schemars` from an input struct), `async fn call(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>`. A `ToolRegistry` holds them by name.
- [ ] 7.2 Extend `llm-core` for tools: `ContentBlock` gains `ToolUse { id, name, input }` and `ToolResult { tool_use_id, content, is_error }`, and `LlmRequest` gains `tools: Vec<ToolSpec>`. Translate `ToolSpec` into Converse's `ToolConfiguration` (name, description, `inputSchema.json`), both streaming and not. ⚠️ When streaming, tool arguments arrive as a JSON string in pieces inside `ContentBlockDelta`: accumulate them and parse only at `ContentBlockStop`.
- [ ] 7.3 💡 **The loop.** In pseudocode:

  ```text
  messages = [user]
  loop (max N iterations, plus token and time budgets):
      resp = llm.complete(system, messages, tools)
      messages.push(resp.message)
      if resp.stop_reason != ToolUse: break
      results = run every ToolUse in the message, in parallel
      messages.push(a user message with ALL the ToolResults, in the same order)
  ```

  ⚠️ Every `tool_result` from one turn goes in a **single** user message; splitting them teaches the model to stop making parallel calls. A failing tool returns a `tool_result` with `is_error = true`, never an exception that breaks the loop.
- [ ] 7.4 **Limits.** Maximum iterations, an overall timeout and a token budget: without them a confused model can loop forever and burn money. 💡 Real agents also gate dangerous tools behind human approval: the loop suspends and returns a state the caller resumes after a yes. The guiding project has no dangerous tool, so you do not build it here; Module 16 has it as an exercise (16.13).
- [ ] 7.5 📚 **Designing good tools.** Read [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents). Practical rules: few tools with sharp boundaries; names and descriptions written for the model (they say *when* to use it and *when not*); small, typed inputs; compact outputs (the result lands in the context and is paid for on every later turn); descriptive errors that tell the model how to correct itself.
- [ ] 7.6 🔭 **Nested traces.** The agent's span (`langfuse.observation.type = "agent"`) contains one generation per iteration and one `"tool"` span per tool execution, with input and output. In Langfuse the trace should read as a tree: iteration 1 → tool A, tool B → iteration 2 → final answer. Add `iterations` and `tool_calls` to the trace metadata.
- [ ] 7.7 🧪 Example tools for the guiding project: `search_knowledge_base(query, top_k)` (for now over an in-memory list with text search; it becomes vector search in Module 8), `get_user_profile(user_id)`, `get_current_date()`. Build a demo where the model decides on its own when to search, and read it in Langfuse.
- [ ] 7.8 **Tests without a network.** Create `FakeLlmClient` in `llm-core`, behind a `test-util` feature so it never ships in the binary: a second implementation of `LlmClient` that returns the responses you script in each test. With it, write tests verifying that: the loop stops at `EndTurn`; a nonexistent tool produces `is_error`; the iteration ceiling fires; two `tool_use` blocks in one message are executed in parallel and returned in a single message.
- [ ] 7.9 💡 **Model Context Protocol (MCP).** This is the standard for exposing tools to a model through a separate server. With the official `rmcp` crate, write a **client** that connects to an MCP server (for example a sample filesystem server, or one of your own exposing the knowledge base) and adapts its tools to your `Tool` trait. The harness then cannot tell local tools from remote ones.

  💡 **How MCP works.** Without a standard, every application writes an adapter for every source of tools, and every tool is rewritten for every application. The Model Context Protocol separates the two sides. An MCP *server* exposes tools (and also resources and prompts) with their names, descriptions and JSON schemas; an MCP *client*, inside your application, discovers them (`tools/list`), calls them (`tools/call`) and hands the results to your code. Messages are JSON-RPC over one of two transports: *stdio*, where the client starts the server as a local child process and talks through its standard input and output, and *streamable HTTP*, for a remote server shared by several applications, with authentication. The model never talks to the server: your harness does, and adapting MCP tools to the `Tool` trait is what makes local and remote tools look the same to the loop. A remote tool is still code you do not control: its description enters your prompt and its output enters your context, so it is also a prompt injection surface (13.2).
- [ ] 7.10 🧪 Measure in Langfuse how many tokens the tool list alone costs (compare a generation with and without tools). With many tools that cost is paid on every turn, which is why on-demand loading techniques exist (Module 16).
- [ ] 7.11 💡 **Structured output through a tool.** Implement the alternative from 6.8: an `emit_result` tool with the schema you want. Compare reliability and latency against `outputConfig.textFormat`. Record the result per model.

**Done when** the generic loop is tested without a network, a remote MCP tool is usable exactly like a local one, the Langfuse trace shows the agent → generation → tool tree, and you have a demo where the model searches the knowledge base when it needs to.

**Self-check.** What happens if a tool returns 50 KB of text? (The context explodes, cost grows on every turn, quality drops.) What are three ways to avoid it? Why must tool arguments be parsed only at the end of the block when streaming?

---

## Module 8 · Context engineering

⏱ 8–10 hours.

**Goal.** Decide explicitly **what** enters the context of each call, **in what order** and **with what budget**: short-term memory, long-term memory, retrieval, cache. And make it visible in Langfuse.

**Why it matters.** Context is a finite resource with diminishing returns. The quality of an agentic application depends more on what you keep out than on what you put in.

### Steps

- [ ] 8.1 📚 Read [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Concepts to note: context rot, "the smallest set of high-signal tokens", just-in-time retrieval, compaction, structured notes, sub-agents as a tool for context isolation.
- [ ] 8.2 💡 **Counting tokens.** Bedrock exposes a `CountTokens` operation (check availability for your model in your region); otherwise use `usage.inputTokens` from real calls to calibrate a local estimate (characters / 3.5 for Italian). Always record both the estimate **and** the real value: the gap tells you how much to trust the heuristic.
- [ ] 8.3 💡 **An explicit budget.** Define a budget for the `GENERATION` call, for example: stable system ≤ 2,500 tokens, compact profile ≤ 400, knowledge base context ≤ 1,500, recent turns ≤ 3,000, plus the current message. Put it in configuration, not in the code. 🔭 Write the actual per-section split into the generation's `metadata`: in Langfuse you can then see where the tokens go when cost rises.
- [ ] 8.4 🧪 **Short-term memory: a token-budgeted window.** Implement an append-only `conversation_log` in SQLite and derive `recent_turns` **from** the log: read turns backwards, accumulating tokens until the budget runs out. Measure in tokens, not in number of turns, so that people who write short messages are not penalized. Unit tests with fake turns.
- [ ] 8.5 🧪 **Compaction.** When the conversation exceeds the budget, summarize the older turns with a call to the cheap model and replace them with a `<previous_summary>` block. Define what the summary must preserve (decisions made, open topics, tone) and verify with a test that the key information survives. 🔭 Compaction is a generation in its own right, with its own versioned Langfuse prompt (`conversation-summarizer`).
- [ ] 8.6 🧪 **Long-term memory: the user profile.** A `UserProfile` struct with dimensions, `confidence` and `last_updated`. The **compact** view injected into `GENERATION` filters by confidence and recency (example rule from the source document: confidence > 0.5 and updated within the last 90 days, or confidence > 0.8 always). The update (`PROFILE_UPDATE`) is asynchronous, uses structured output, and is **atomic**: it either replaces the whole profile or touches nothing.
- [ ] 8.7 💡 **Retrieval (RAG), the essentials.** Three pieces: embeddings (on Bedrock: Amazon Titan Embeddings or Cohere Embed, through `invoke_model`), a vector store (locally `pgvector` on PostgreSQL, or LanceDB/Qdrant), and a search function with `top_k` and metadata filters. Document chunking matters more than the embedding model: small chunks with metadata (source, review date, author). 🔭 Retrieval is a `retriever` span with `query`, `top_k`, and the ids and scores of the records found.

  💡 **How vector search works.** An embedding model turns a piece of text into a vector, a list of a few hundred to a few thousand numbers, trained so that texts with similar meanings give vectors pointing in similar directions. Similarity is measured by the angle between two vectors (cosine similarity) or by their dot product. Searching then means embedding the query with the same model and returning the `top_k` stored vectors closest to it; a vector store keeps the vectors in an index that finds the nearest ones without comparing the query against all of them (approximate nearest neighbour), trading a little precision for speed. Three consequences follow.
  - **The search is by meaning, not by words.** "Mi fa male la testa" can find a record about headaches, while a code or a proper name is often matched better by plain keyword search: that is why many systems combine the two.
  - **Chunking decides what can be found.** A chunk that mixes three topics has a vector that represents none of them well, and a chunk that is too small loses its context. Small, self-contained chunks with metadata usually win.
  - **Queries and documents must be embedded alike**: with the same model, and in the same language where possible.

  *Recall@3* (8.8) is the share of test queries for which the right record is among the first three results.

  💡 **Why retrieve at all: grounding.** A model generates the most plausible continuation, not a verified fact: when it lacks the information, it can produce a fluent, confident and wrong answer, a *hallucination*. Putting the relevant records in the context and asking the model to answer only from them, citing them, makes the answer *grounded*, that is, checkable against a source. It reduces invention without removing it, which is why `GENERATION` returns the ids of the records it used and the evaluation checks that every claim is supported by them (`grounded_in_kb`, 11.2).
- [ ] 8.8 🧪 Replace the in-memory knowledge base from Module 7 with the vector one. Use 20–30 synthetic records. Measure *recall@3* over 15 hand-written queries. Then try the managed alternative, **Bedrock Knowledge Bases**, to see what work it removes and what control it takes away.
- [ ] 8.9 💡 **Order and cache.** Recompose the `GENERATION` prompt like this: [stable system + schema + rules] → `cachePoint` → [compact profile] → [knowledge base context] → [recent turns] → [message]. Verify `cache_read_tokens` across consecutive calls from the same user: in Langfuse it appears in the generation's usage details and, if you configured the cache read price in 5.4, in the cost.
- [ ] 8.10 💡 **Cleaning up agentic context.** In loops with many tools, old results become ballast: replace the oldest `tool_result` values with a placeholder ("result removed, 1,240 tokens") after K turns. Measure the effect on cost and quality.
- [ ] 8.11 Write `docs/adr/000x-context-budget.md`: budget per section, compact view rules, compaction strategy, rationale.

**Done when** the budgeted window, compaction and the compact profile are tested without a network, the vector knowledge base works, the cache is read on consecutive calls, and generation metadata in Langfuse shows the token split.

**Self-check.** Why must `recent_turns` be derived from the log rather than written separately? (One source of truth, no dual-write inconsistency.)

---

## Module 9 · The guiding project's pipeline

⏱ 10–14 hours.

**Goal.** Compose the pieces into a working end-to-end pipeline, available from the CLI and over HTTP, with one Langfuse trace per message, and with streaming plus parallel guardrails as a separate **extension**.

**Why it matters.** This is where the concepts become a system: concurrency, cancellation, retries, atomicity, and the design questions no prompting tutorial asks you.

### Steps

- [ ] 9.1 💡 Model the pipeline as an **explicit workflow**, not a free-running agent: a `Stage` enum and one function per step, each with typed input and output. The code decides the flow; the LLM works inside the steps. That is the right pattern when the flow is known in advance, and it makes every step testable on its own.
- [ ] 9.2 Implement `STATE_READ` → `SAFETY` (with `RETRIEVAL` in parallel through `tokio::join!`, discarding its result if `SAFETY` blocks) → `GENERATION` → `GUARDRAILS` → `STATE_WRITE` → reply → `PROFILE_UPDATE` in the background (`tokio::spawn`, with tracing linked to the request's span).
- [ ] 9.3 🔭 **One trace per message.** The root span is the trace, with `session.id` as the conversation, a pseudonymized `user.id`, `tags = ["pipeline", environment]`, `input` as the user message and `output` as the final reply. One child span per step, named after the step; LLM calls are generations inside the step. `PROFILE_UPDATE` runs after the reply: make it either a delayed child span **or** a separate trace with `metadata.parent_trace_id`; choose and document. Every prompt used carries its name and version (6.6).
- [ ] 9.4 💡 **Escalation.** The `SAFETY` escalation replies are **fixed texts** decided by people, not generated: in a healthcare domain that is a safety and accountability choice. Put them in configuration. 🔭 Tag the trace `escalation:<type>`.
- [ ] 9.5 💡 🔭 **The evaluator-optimizer loop.** If `GUARDRAILS` fails, re-run `GENERATION` with the failure reason in the prompt; at most two attempts, then a safe fallback reply. Record the outcome as a **score** on the trace (`guardrail_pass` boolean, `guardrail_failure_type` categorical) and the attempt count in metadata: failure frequency becomes a chart and an alert in Module 14.
- [ ] 9.6 💡 **Robustness.** A timeout per step; retries with backoff and jitter only on retryable errors; idempotent writes (a repeated request must not duplicate turns); an atomic `PROFILE_UPDATE` with rollback; clean cancellation if the client disconnects (`CancellationToken`). Errors become spans with `level = ERROR` and a `status_message`, so you can filter them in Langfuse.

  💡 **Idempotency and atomicity.** An operation is *idempotent* when repeating it has the same effect as doing it once. It matters because retries, timeouts and a client that resends after losing the connection all repeat requests: if saving a turn is not idempotent, the conversation gets the same message twice. The usual technique is a key chosen by the caller (a request id) plus a uniqueness constraint in the database, so the second insert is recognized and ignored. An operation is *atomic* when it either happens entirely or not at all: the profile update writes several fields, and an error halfway must not leave some old values and some new ones, which a database transaction guarantees (the writes commit together or roll back together). Cancellation makes both necessary: in async Rust a cancelled future stops at its last `.await` (4.8), so any work spread over several `.await`s can be interrupted in the middle.
- [ ] 9.7 Expose the pipeline over HTTP with `axum`: `POST /chat` (full reply) and `POST /chat/stream` (SSE). Fake authentication for now (a header carrying `user_id`).

  💡 **How SSE works.** Server-Sent Events is plain HTTP in which the server does not close the response: it answers with `Content-Type: text/event-stream` and keeps writing events as they happen, each a few `field: value` lines (`event:`, `data:`, `id:`) followed by a blank line. It flows one way, server to client, which is exactly the shape of a streamed reply, and it passes through proxies and load balancers that understand HTTP; WebSockets are two-way and more complex than this needs. In the browser, `EventSource` reads SSE and reconnects on its own, but it only makes GET requests, so a `POST /chat/stream` is read with `fetch` and a stream reader that parses the same format. Two things to watch: a proxy that buffers responses turns the stream back into a single block, and an idle timeout can cut a long reply, which a periodic comment line (`: keep-alive`) prevents.
- [ ] 9.8 🧪 Measure per-step latency in Langfuse over 20 synthetic messages: `SAFETY`, `GENERATION` (time to first token and total), `GUARDRAILS`. Compare against the estimates in the source document. Record cost per message (Langfuse sums it per trace).
- [ ] 9.9 🧪 **Extension: streaming with parallel guardrails.** Forward `GENERATION` tokens to the client as they arrive while `GUARDRAILS` analyzes the accumulated text (in chunks, or at end of stream). On failure, send a `retract` event followed by the corrected reply. ⚠️ This is an epic of its own: buffering, clean aborts, and the user experience of text disappearing. Do it only after the synchronous version is stable and measured, and document the trade-off in an ADR.
- [ ] 9.10 💡 **The agentic alternative.** In a branch, rewrite `GENERATION` as an agent with tools (`search_knowledge_base`, `get_user_profile`) instead of pre-injected context. Compare quality, cost, latency and predictability in Langfuse. This comparison is the central lesson of "workflow versus agent".
- [ ] 9.11 ADR: pipeline structure, concurrency choices, retry policy, the workflow-versus-agent decision, the shape of the trace.

**Done when** `POST /chat` answers end to end on synthetic data, every message produces a Langfuse trace with one span per step and guardrail scores, the profile updates in the background without corrupting itself on error, and you have a table of per-step latency and cost.

**Self-check.** Why can `RETRIEVAL` start in parallel with `SAFETY` while `GENERATION` cannot? What makes an update "atomic" in your code?

---

## Module 10 · Model router

⏱ 8–10 hours.

**Goal.** A `router` crate that picks the model for each call according to explicit rules, handles fallback and degradation, and surfaces its decisions in Langfuse.

**Why it matters.** Every step has different needs; a classification does not need the biggest model. The router is where cost, latency, quality and availability meet. Done badly it hides problems; done well it is the cheapest optimization lever you have.

### Steps

- [ ] 10.1 💡 **ModelRegistry.** A table of `ModelSpec` entries loaded from configuration (TOML): `id`, `provider`, `tier` (small/mid/large), input/output/cache price per Mtok, context window, `supports_tools`, `supports_structured_output`, `supports_streaming`, `accepted_params` (which of `temperature`, `top_p`, `effort` and `thinking` the model takes; it replaces the capability table from 6.4), `region_profile`, quotas (rpm, tpm), measured p50 latency. You **verify capabilities with tests** (Module 12), you do not copy them from the docs. The prices must match the ones entered in Langfuse (5.4): keep them in one file and generate both.
- [ ] 10.2 💡 **Routing strategies**, in increasing order of complexity:
  1. **Static per step**: `SAFETY → small`, `GENERATION → mid/large`, `GUARDRAILS → small`, `PROFILE_UPDATE → small`. This covers 90% of real cases. The prompt's `config` in Langfuse (6.4) names the **role** (`fast`, `main`) and the router translates a role into a model, so changing the model does not mean touching the prompt, and vice versa.
  2. **Rule-based**: on input length, language, presence of tools, structured output requirements, premium users.
  3. **Cascading**: try the cheap model; if its stated confidence is low or the output fails schema validation, escalate to the bigger one. Measure how often it escalates. ⚠️ A confidence the model states about itself is poorly calibrated: models often sound as sure when they are wrong as when they are right. Prefer signals you can check (schema validation, an explicit `UNSURE` label defined in the prompt, agreement between two cheap calls), and judge the cascade on its measured accuracy, not on the number the model reports.
  4. **Classifier-driven**: a small model (or a rule) estimates difficulty and picks the tier. Mind the cost of the extra call.
  5. **Managed**: *Amazon Bedrock Intelligent Prompt Routing* picks among models in a family based on the prompt. Try it to see what it does, and compare it with your strategy 3.
- [ ] 10.3 💡 **Fallback and resilience.** A chain per role: primary → secondary (same tier, different model or a different EU region) → degradation (a courtesy reply, a queue). Retry only on retryable errors, with backoff and jitter (the SDK does this for a single call; the router does it **across models**). A circuit breaker per model: after N errors in T seconds, skip to the secondary for a while, then probe again gradually.

  💡 **How a circuit breaker and a token bucket work.** A *circuit breaker* stops calling a model that is failing, instead of paying a timeout on every request. It has three states: *closed*, where calls pass and failures are counted; *open*, entered after N failures in T seconds, where calls go straight to the fallback for a cooldown period; and *half-open*, when the cooldown ends, where a few trial calls pass, success closes the circuit and failure opens it again. It protects your latency and gives the failing service room to recover. A *token bucket*, the mechanism behind `governor` (10.4), limits your own rate before the provider has to: a bucket holds up to B tokens and refills at R per second, each request takes one (or, against a tokens-per-minute quota, as many as it will use), and a request that finds the bucket empty waits or goes elsewhere. B allows short bursts and R is the sustained rate: set them below the quotas from 3.7, and throttling from the provider becomes rare.
- [ ] 10.4 💡 **Constraints to respect.** The request's remaining timeout (falling back makes no sense with 200 ms left); per-model quotas (a local token bucket with `governor` keeps you out of throttling); region (never leave the EU); capability (do not route a request with tools to a model that does not support them).
- [ ] 10.5 ⚠️ **The cache is per model.** Switching models mid-conversation invalidates the prompt cache and can change style and behaviour. The router should prefer **per-session stability**: pin the model at the start of a conversation and change it only on errors, not for instantaneous optimization.
- [ ] 10.6 🔭 **The router in Langfuse.** Every generation carries `router.role`, `router.reason` (`static`, `fallback:throttled`, `escalation:low_confidence`) and `router.attempt` in its metadata; the model actually used is already in the model field. In Langfuse build a **dashboard** with: generation distribution per model, cost per role, fallback counts (filtered on metadata). You will see immediately if a cheap model escalates too often.
- [ ] 10.7 💡 **Controlled experimentation, as a concept.** Two techniques build on the router, and each is built when something needs it. A per-request *override* forces a specific model: you add it in Module 11, when the evaluation runner has to choose the model under test (11.5). *Shadow routing* also sends a sample of real requests to a candidate model in the background, without affecting the reply: it needs production traffic, and it appears in Module 16 as part of migrating to a new model (16.10).
- [ ] 10.8 🧪 Simulate failures with `FakeLlmClient` (a burst of throttling, timeouts, invalid output) and verify: correct fallback, a circuit breaker that opens and closes again, and no request exceeding the configured quota.
- [ ] 10.9 🧪 With real models: compare the static strategy against the cascade over 50 synthetic messages. Total cost, p50/p95 latency, and quality measured with the `SAFETY` dataset script from 6.10; Module 11 redoes this comparison with proper statistics. Write the ADR with the decision.

**Done when** the router is configured from a file, tested against failures without a network, and every decision is readable in Langfuse with its reason.

**Self-check.** Why is "the cheapest model that passes the tests" the wrong rule for `GENERATION` when the cost per **completed task** (not per call) is higher because of retries and regenerations?

**Going deeper.** Bedrock's [Intelligent Prompt Routing](https://aws.amazon.com/bedrock/intelligent-prompt-routing/) page; the orchestrator-workers pattern from [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), for delegating sub-tasks to small models without changing the model in the main loop.

---

## Module 11 · Evaluating models with datasets and experiments

⏱ 10–12 hours.

**Goal.** A runner in `evals` that executes a **Langfuse dataset** with a configuration (model, prompt@version, effort), records results as a **dataset run** with scores, and produces a comparable report: quality with a confidence interval, cost, latency. Plus Langfuse-managed LLM judges and calibration against human annotations.

**Why it matters.** Without evaluation you decide by feel. With a **broken** evaluation you decide confidently in the wrong direction. Most "surprising results" are bugs in the evaluation.

### Steps

- [ ] 11.1 📚 🔭 Read [LLM Evaluation Concepts](https://langfuse.com/docs/evaluation/core-concepts) and [Evaluation Overview](https://langfuse.com/docs/evaluation/overview) in the Langfuse docs. Fix these: **dataset** (a collection of items with `input`, `expected_output`, `metadata`), **dataset run** (an experiment: each executed item produces a trace linked to the item, plus scores), **evaluator** (an LLM judge or a function assigning scores to production or experiment traces), **annotation queue** (a queue for human labelling), **score** in its three types (numeric, categorical, boolean).
- [ ] 11.2 💡 **What you evaluate, per step.** `SAFETY`: accuracy, and above all **false negatives** (a risk that went unseen). `GENERATION`: schema conformance, correct use of knowledge base references (no claims absent from the cited records), tone, Italian language quality, absence of diagnosis. `GUARDRAILS`: agreement with human judgement. `PROFILE_UPDATE`: extraction correctness. Each step gets its own dataset.
- [ ] 11.3 🔭 **Datasets.** Source of truth: JSONL files under `datasets/`, versioned in git. Add a `just datasets-push` recipe that uploads them to Langfuse as datasets (`safety-v1`, `generation-v1`…) through the API. Where the cases come from: written by hand, synthesized with a model **and then reviewed by a person**, or (in production) from real traces added to the dataset with one click in the Langfuse interface. ⚠️ Never use the output of the model under evaluation as ground truth. Balance the classes. For `SAFETY`, include cases in both directions explicitly (must fire / must not fire).
- [ ] 11.4 💡 **Types of grader.** (1) Programmatic: exact match, schema validation, regex, checking that cited knowledge base ids exist. (2) Rubric with an LLM judge: **atomic** properties scored one at a time (`no_diagnosis`, `grounded_in_kb`, `tone_ok`), with structured output, treating the candidate text as data rather than instructions. (3) Pairwise: two answers, which is better (randomize the A/B order). Use a judge from a **different family** than the model under evaluation when you can.

  💡 **Why a judge needs care.** A model judging text has systematic biases: it tends to prefer the longer answer, the one shown first in a pairwise comparison, the more confident tone, and text written by its own model family; when it is the same model as the one under evaluation, it also shares its blind spots. The rules in this step answer those biases one by one: atomic yes/no properties instead of a single 1–10 score, which leaves less room for vague preference; a rubric with explicit criteria; random A/B order; a different model family. Above all, calibration against human labels (11.7): a judge is a measuring instrument, and an instrument never checked against a reference measures nothing you can trust.
- [ ] 11.5 🔭 **The runner as a dataset run.** Create the `evals` crate and a `just eval` recipe. To pick the model under test, add to the router a per-request `override` that bypasses its rules and is recorded as `router.reason = override`; production reuses it as an emergency switch (15.12). `just eval --dataset safety-v1 --model X --prompt safety-classifier@v3 --reps 3` does this: create a run with a descriptive name (`safety-v1 / haiku / v3 / 2026-09-20`); for every item and repetition, execute the step's real pipeline (not a copy); link the trace to the dataset item; compute the programmatic graders and write them as scores on the run. Run metadata: model, prompt@version, effort, git commit.
- [ ] 11.6 🔭 **Langfuse-managed judges.** Under *Evaluation → Evaluators* create an LLM-as-a-judge evaluator for `no_diagnosis` and one for `grounded_in_kb`, with your rubric, connected to Bedrock, filtered to experiment traces. Langfuse runs them by itself on every new matching trace. Compare with the alternative, a judge inside the runner: the former is convenient and reusable in production (Module 14), the latter is testable and versioned in git. You can keep both.
- [ ] 11.7 🔭 **Calibrating the judge.** Send 30–50 traces to an **annotation queue**, label them yourself (or with a clinical colleague) using the same scores the judge assigns, then compare judge-versus-human agreement. Below 90% on clear-cut cases, the rubric needs another pass. Repeat whenever you change the judge model.
- [ ] 11.8 💡 **Performance metrics from the API, not estimated.** Input, output and cache tokens from the response; cost computed from the price list of the model **that actually answered**; latency of the successful call only, excluding retries. The judge's cost is visible separately in Langfuse because its generations are distinct traces.
- [ ] 11.9 💡 **Repetitions and noise.** Run every case R times (at least 2–3). Over N independent trials, the half-width of the 95% confidence interval on a success rate is at most about `1/sqrt(N)` (6.10). Repetitions of the same case are not independent, though: a hard case tends to fail every time. So with n cases and R repetitions the real margin lies between `1/sqrt(n·R)` and `1/sqrt(n)`: 30 cases × 2 repetitions gives between ±13 and ±18 points. Repetitions measure how much the model varies on each case; only more cases narrow the margin on the dataset. If the difference between two runs is below the noise, **there is no difference**. The runner prints that number in the report; Langfuse shows per-run averages, not intervals, so compute them yourself.
- [ ] 11.10 💡 **Evaluation harness hygiene.** Separate infrastructure errors (timeouts, throttling, output truncated at `max_tokens`) from model errors: the former become an `infra_error = true` score and never enter the quality average. The full trajectory is already in Langfuse (the trace linked to the item). Verify that the model that answered is the one requested. Run an **oracle** (the expected answers must pass) and a **null baseline** (an empty answer must fail): if either misbehaves, the evaluation is broken.
- [ ] 11.11 🧪 🔭 **Model comparison for `SAFETY`.** Enable a non-Anthropic model first, Amazon Nova or an open-weight one such as Llama or Mistral (Console → Bedrock → *Model catalog*, no form needed), and add it to the `ModelRegistry`. Then three runs (Haiku, Sonnet, the non-Anthropic model) on the same dataset and prompt. Use Langfuse's **run comparison** view for the dataset: average scores, cost and latency side by side. Then open the cases where the runs disagree: they are the most instructive. Decision in an ADR.
- [ ] 11.12 🧪 **Comparison for `GENERATION`.** A mixed grader: programmatic schema and grounding checks, an LLM-judge rubric for tone and absence of diagnosis. Also try the same model at different `effort`: it is often the cheapest variable.
- [ ] 11.13 🔭 **Experiments from the interface.** Langfuse can launch an experiment on a dataset directly from the UI (prompt × model) with no code. Try it on `safety-v1`: it is useful for iterating on prompts without touching the runner, but it does not execute **your** pipeline (schema, router, tools). Note when to use which.
- [ ] 11.14 💡 Also try **Amazon Bedrock Evaluations** (an AWS-managed LLM judge) to see what the provider offers. Compare with Langfuse: where the data lives, how much control you have over the rubric, how well it integrates with the real pipeline.
- [ ] 11.15 💡 **The evaluation is alive.** Every bug found in production becomes an item (in Langfuse: trace → "add to dataset"). Every item on which all models score 100% is made harder or retired. Recalibrate the judge whenever you change the judge model.

**Done when** datasets live in git and in Langfuse in sync, the runner produces dataset runs with scores and a report with confidence intervals, a managed evaluator runs on experiment traces and is calibrated against human annotations, and you have two ADRs recording the model choice for `SAFETY` and `GENERATION`.

**Self-check.** If run A scores 92% and run B scores 88% over 25 cases with one repetition, what can you conclude? (Nothing: the noise is ±20 points.) Why must the runner call the real pipeline rather than a simplified copy?

---

## Module 12 · Regression test suite

⏱ 8–10 hours.

**Goal.** A test pyramid that protects the harness (deterministic) and the model integration (non-deterministic) with different tools at different cadences, using Langfuse as the record of regression experiment results.

**Why it matters.** Models change under your feet (new versions, provider behaviour changes), prompts change by your hand (and now also from a web interface), and the harness changes through refactoring. You need different nets.

### Steps

- [ ] 12.1 💡 **Level 1: harness unit tests (per pull request, no network).** Everything that is pure logic: prompt rendering, structured output parsing, the token-budgeted window, compact view rules, router decisions, the tool loop, `PromptStore` fallback. Use `FakeLlmClient` and a `FakePromptStore`. There should be hundreds of them and they should run in seconds with `cargo nextest`.
- [ ] 12.2 💡 🔭 **Prompt snapshot tests.** With `insta`, a snapshot of the **rendered** prompt for each step with fixed inputs, built from the `prompts/` snapshots downloaded from Langfuse (6.5). If someone moves the `production` label in Langfuse, the next `just prompts-pull` changes the file, the snapshot fails, and the change appears in a pull request diff: that is your review gate on prompts edited from the interface.
- [ ] 12.3 💡 **Level 2: Bedrock adapter tests (per pull request, no network).** With `aws-smithy-mocks` (or the SDK's `StaticReplayClient`) you simulate service responses and verify type mapping, streaming event handling, error classification (throttling, validation, access denied) and retry behaviour. Record real responses once and use them as fixtures. Same approach for the Langfuse client with `wiremock`.
- [ ] 12.4 💡 **Level 3: contract tests against the real services (nightly, `#[ignore]`).** A handful of tests, one per capability declared in the `ModelRegistry`: the model answers; it supports tools; it supports structured output; cache reads work; streaming emits the expected events; a small `max_tokens` produces `stop_reason = max_tokens`; every parameter in `accepted_params` is accepted, and one it omits (such as `effort` on Haiku 4.5) is rejected when sent directly, bypassing the capability table, so you find out the day a provider changes the rules. Plus two for Langfuse: a trace sent over OTel appears through the API within N seconds, and a `production` prompt downloads. They run with `cargo nextest run --run-ignored only`. This is the first time CI needs real credentials: create an IAM role that GitHub Actions assumes through OIDC, allowed only to invoke your inference profiles, and store the Langfuse keys for CI as repository secrets. Never put static AWS access keys in secrets.

  💡 **How OIDC federation works.** Instead of storing an AWS key in GitHub, you tell AWS to trust GitHub as an identity provider. For every job, GitHub issues a short-lived signed token (an OpenID Connect token) stating who is asking: repository, branch, workflow. The job presents it to AWS STS (`AssumeRoleWithWebIdentity`), which verifies the signature and the role's trust policy, for example "only this repository, only the `dev` branch", and returns temporary credentials valid for about an hour. Nothing long-lived exists that could leak, and the trust policy decides which repository and branch may use the role.
- [ ] 12.5 💡 🔭 **Level 4: regression experiments (nightly, or on prompt and model changes).** The nightly job runs the Module 11 runner over the regression datasets, then reads the run's scores through the API and compares them against **thresholds**: `safety.false_negative_rate ≤ 2%`, `generation.schema_valid ≥ 99%`, `generation.grounded ≥ 95%`, average cost per message ≤ X. The threshold must sit **above** the estimated noise, otherwise the test is a false-alarm generator. Failure means a red build, with a link to the Langfuse run in the message. A Langfuse **alert** on the run's average score (Module 14) is the second net.
- [ ] 12.6 💡 **Non-determinism.** In level 3 and 4 tests never assert on exact text: assert properties (valid schema, field present, value within a set), over several repetitions, above a threshold. Mark the tests prone to flakiness and track their failure rate: if it rises, that is a signal, not an annoyance.
- [ ] 12.7 💡 **Pinning and canaries.** Pin model versions where the provider allows it; when a new version arrives, run the full level 3 and 4 suite against it **before** changing the registry (that is your canary): in Langfuse the new run is compared against the last good one. Document the migration in an ADR.
- [ ] 12.8 Property-based tests with `proptest` for the hostile functions: the budgeted window with randomly sized turns, the structured output parser with almost-valid JSON, approximate tokenization.
- [ ] 12.9 Security as regression: an `injection-v1` dataset of prompt injection attempts that **must not** change `GENERATION` behaviour or bypass `SAFETY` (Module 13). It is a Langfuse dataset like any other.
- [ ] 12.10 Wire it into CI: `ci.yml` already runs levels 1–2; create `nightly.yml` now, which runs levels 3–4 on a schedule and publishes the report with links to the runs; add a badge to the README.

**Done when** a prompt change in Langfuse makes a snapshot fail in a pull request after the pull; a throttling mock is handled correctly; the nightly job runs against the real services, produces Langfuse dataset runs and compares them against thresholds.

**Self-check.** Why is a level 4 test with a 95% threshold over 20 cases and one repetition a useless test? What protects you from someone moving `production` onto an untested prompt?

---

## Module 13 · Security, privacy and guardrails

⏱ 6–8 hours.

**Goal.** Layered defences against hostile input, data leakage and out-of-scope behaviour; baseline compliance for healthcare data in the EU, including the data that ends up in Langfuse.

**Why it matters.** An agentic application has a new attack surface: user input and retrieved documents are text the model might read as instructions. In healthcare a mistake is real harm. And an observability tool, by its nature, **copies** data: it must be governed like the system it observes.

### Steps

- [ ] 13.1 📚 Read the [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/). For each of the ten entries, write whether and how it touches the guiding project.
- [ ] 13.2 💡 **Prompt injection.** Defences: clear delimitation of data (XML tags), system prompt instructions to treat that content as data, least privilege for tools (a tool cannot do more than it needs to), human confirmation for irreversible actions, output validation before execution or display. No defence is complete: hence the layers and the regression dataset (12.9).
- [ ] 13.3 💡 **Bedrock Guardrails.** Create a guardrail with content filters, denied topics (for example "diagnosis"), a PII filter with masking, and a grounding check (is the output supported by the source?). Apply it with `ApplyGuardrail` to input and output as a layer **on top of** your own LLM `GUARDRAILS`. Compare: what does one catch that the other does not? What cost and latency does it add? 🔭 The outcome is one more score on the trace (`bedrock_guardrail_action`).
- [ ] 13.4 💡 🔭 **Personal data in prompts and in traces.** Minimize what enters the prompt; pseudonymize identifiers (the `user.id` in Langfuse must never be a real identifier); apply **PII redaction before export** in traces (a function in `observability`, switched on per environment through configuration, covering names, phone numbers, emails and national identifiers; for free text consider a small model or the Bedrock Guardrails PII API); set **retention** per Langfuse project (Pro plan or above, 5.10); document the data flow (who sees what, in which region). Bedrock does not use your data to train models and does not retain it beyond the request, except for logging you enable yourself: revisit the choice from 3.9.

  💡 **Health data under the GDPR, the essentials.** Health data is a *special category* of personal data (Article 9): processing it needs an explicit legal basis and stronger safeguards. *Pseudonymized* data, where identifiers are replaced by codes but whoever holds the key can link them back, is still personal data; only truly *anonymous* data, which nobody can reasonably re-identify, falls outside the regulation, and the free text of a conversation is rarely anonymous. Every service that processes data on your behalf, Bedrock and Langfuse included, is a *processor* or sub-processor that needs a data processing agreement and a known location, which is why data residency (13.5) and retention matter. *Minimization* is the principle that simplifies everything else: data you do not collect, send or store needs no protection. This is orientation, not legal advice: a real product needs a review by the company's data protection officer.
- [ ] 13.5 💡 🔭 **Data residency.** Only `eu.*` inference profiles; no feature that routes outside the EU; Langfuse Cloud EU region (5.2); verify where logs, traces, datasets and exports end up. Write it all in an ADR: it will be the basis of the legal review.
- [ ] 13.6 💡 **Least-privilege IAM.** Replace `AmazonBedrockFullAccess` with a policy scoped to the ARNs of the inference profiles you use; apply it to your own identity and to the CI role from 12.4; the deployed app gets its own role in Module 15 (15.5). No static access keys anywhere.
- [ ] 13.7 💡 🔭 **Secrets and Langfuse access.** separate Langfuse API keys for your machine, CI and, from Module 15, the deployed app, which reads them from Secrets Manager; in Langfuse, members with minimal roles (who can move `production` on prompts, who can see traces with content); the `production` label **protected**.
- [ ] 13.8 💡 **Abuse and cost.** Per-user rate limiting (`governor` in axum), a maximum message length, a cost ceiling per session and per day, alerts on anomalies (one user generating 30% of the cost: the Langfuse *Users* view shows it).
- [ ] 13.9 💡 **Output to the user.** Escalation replies are fixed (9.4). Generated replies pass through `GUARDRAILS`, through Bedrock Guardrails and through programmatic validation (length, language, no URLs outside an allowlist).
- [ ] 13.10 ADR: threat model, controls per layer, what remains uncovered and why.

**Done when** the injection dataset passes, Bedrock Guardrails is integrated and measured, IAM is least-privilege, `prod` traces are redacted, retention is configured, and the data ADR is written.

**Self-check.** Why is "I tell the prompt to ignore instructions found in documents" not enough as a defence? Why must your observability tool be treated as a system that processes personal data?

---

## Module 14 · Production observability with Langfuse

⏱ 8–10 hours.

**Goal.** Langfuse as the operations centre: every request is a trace with a span per step; dashboards and alerts on the metrics that matter; **online** evaluation with managed judges over a sample of traffic; user feedback as scores; annotation queues for human review; correlation with the rest of the infrastructure.

**Why it matters.** In production you cannot read every conversation. You need to know what it costs, how slow it is, how often guardrails fire, when a model changes behaviour, and be able to reconstruct one problematic conversation in minutes.

### Steps

- [ ] 14.1 💡 **What to observe.** Per request: `trace_id`, `session_id`, pseudonymized `user_id`, step, chosen model and reason, prompt name and version, input/output/cache tokens, cost, latency (time to first token and total), `stop_reason`, guardrail outcome, classified errors, tools called with their duration. Aggregates: cost per day and per role, p50/p95 latency per step, error rate per model, guardrail failure rate, router fallback rate, cache hit rate, model distribution, online quality scores. Almost all of it is already in the traces from Modules 5–10; here you make it legible.
- [ ] 14.2 🔭 **Environments and releases.** Every trace carries `environment` (`prod`, `staging`) and `release` (the app version or commit). Every chart then filters by environment, and every metric change lines up with a deployment.
- [ ] 14.3 🔭 **Dashboards.** Under *Dashboards*, build a "Pipeline" dashboard with: daily cost per router role; p95 latency per step (filtered on span name); trace count by `escalation:*` tag; `guardrail_pass` rate (the average of a boolean score is the percentage of true values); fallback rate (metadata `router.reason` starting with `fallback`); cache reads as a share of total input; generation distribution per model. Add the **Pulse** view on the observations table to spot cost and latency outliers.
- [ ] 14.4 🔭 **Alerts.** In Langfuse create threshold alerts with Slack or webhook notification on: average `guardrail_pass` below threshold over the last hour (a sudden jump often means the provider changed the model, or someone moved `production` onto a different prompt); daily cost above budget; `GENERATION` p95 above threshold; error rate per model. Link evaluator alerts (14.6) directly from the evaluator page.
- [ ] 14.5 🔭 **Reconciled costs.** The cost Langfuse computes (from tokens and the prices in 5.4) should be compared monthly against the AWS bill. To make the bill readable, create now one **Application Inference Profile** per environment, tagged `project=agentic-playground`, route the app's calls through it, and activate that tag under *Cost allocation tags*: Cost Explorer then shows cost per profile. If they do not match within 5%, a price or a model id in the registry is wrong.
- [ ] 14.6 🔭 **Online evaluation.** The LLM-as-a-judge evaluators from Module 11 also run in production: configure them with a filter (`environment = prod`, sampling at say 5%, excluding irrelevant steps) so Langfuse assigns `no_diagnosis` and `grounded_in_kb` to a sample of real traffic. The judge's cost is measurable and must be kept in check through sampling. Also add **code-based** evaluators (for example: the reply cites only knowledge base ids that exist) where a model is not needed.
- [ ] 14.7 🔭 **User feedback.** The client sends thumbs up or down (and an optional comment) together with the reply's `trace_id`, which `POST /chat` starts returning in its response body now, because this is the first thing that needs it; the app writes it as a `user_feedback` score on the trace through the API. In Langfuse, filter traces with negative feedback: they are the primary source of new dataset items (11.15).
- [ ] 14.8 🔭 **Continuous human review.** An annotation queue, "weekly clinical review", fed automatically with traces where `guardrail_pass = false`, where feedback is negative, or where the judge's score is below threshold. A person labels them; comparing human labels against the judge keeps the judge calibrated over time (11.7).
- [ ] 14.9 💡 **Drift.** Weekly, compare online scores and distributions (reply length, escalation rate, cost per message) against the previous release's baseline: a change with no deployment of your own is almost always a change in the model or in a prompt. Comparing dataset runs (12.7) confirms or refutes it.
- [ ] 14.10 💡 **The rest of the infrastructure.** Langfuse specializes in the LLM; CPU, memory, databases, queues and HTTP latency live elsewhere (in this company: Datadog). Two ways to correlate: (a) use the **OpenTelemetry Collector** as a fan-out, with the app sending traces to the collector and the collector forwarding to both Langfuse and Datadog (the same traces, with identical `trace_id`, in two places); (b) export aggregate metrics from Langfuse through the **Metrics API** into the existing monitoring system. Try (a): it is the pattern that lets you change backends without touching the app.
- [ ] 14.11 💡 🔭 **Langfuse volume and cost.** Estimate traces per month and observations per trace; check the Cloud plan's limits; set retention, which needs the Pro plan or above (5.10); use **batch export** (to S3) for historical offline analysis.
- [ ] 14.12 Write a short runbook: "latency is rising" → what to look at in Langfuse and Datadog; "guardrails fire more often" → compare prompt version and model over the last hours; "cost has doubled" → cost-per-role dashboard, then *Users*; "the online score is dropping" → last regression run, last deployment, last label move.
- [ ] 14.13 🧪 Try debugging one conversation: from a `trace_id`, reconstruct every step, the exact prompts (with versions, clickable from the generation), the router's decisions and the cost. If you cannot do it in five minutes, something is missing from the instrumentation.

**Done when** a "Pipeline" dashboard exists with active alerts, an online evaluator runs on a sample of `prod`, user feedback arrives as scores, an annotation queue fills itself, and the runbook is written.

**Self-check.** Why is the `guardrail_pass` rate an excellent canary for silent provider changes? Why is sampling essential for online evaluation?

**Going deeper.** Langfuse docs: [LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge) (evaluators on live traffic), [Custom Dashboards](https://langfuse.com/docs/metrics/features/custom-dashboards), [Alerts](https://langfuse.com/docs/observability/features/alerts), [Annotation Queues](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues), [Metrics API](https://langfuse.com/docs/metrics/features/metrics-api), [Public API](https://langfuse.com/docs/api-and-data-platform/features/public-api).

---

## Module 15 · Deploying to production

⏱ 8–12 hours.

**Goal.** The application running on AWS with IAM roles rather than keys, per-environment configuration, working streaming, gradual rollout, cost controls; plus Langfuse Cloud reviewed for production traces.

**Why it matters.** Many of the choices made so far (streaming, timeouts, caching, quotas, trace volume) can only be measured with a real deployment and some load.

### Steps

- [ ] 15.1 **Packaging.** A multi-stage Dockerfile for Rust (build in a toolchain image, run on `distroless` or `debian-slim`), an optimized binary (`--release`, LTO), a small image. Vulnerability scanning (`cargo deny`, an image scanner).
- [ ] 15.2 💡 **Twelve-factor configuration.** Everything from environment variables or Parameter Store: region, model ids per role, context budgets, router thresholds, Langfuse host and keys, tracing policy (content, sampling). No `if env == prod` in the code.
- [ ] 15.3 💡 **Where to run the app.** Evaluate three options and choose with an ADR: **ECS Fargate** (an always-on HTTP service, natural SSE streaming, the sensible default); **AWS Lambda** with `cargo-lambda` (great for spiky load; ⚠️ response streaming, timeouts and **flushing OTel traces** before the invocation ends all need verification); **Bedrock AgentCore Runtime** (a managed environment for agents, with a Gateway for tools, Memory, Policy with Guardrails and built-in Observability: useful for seeing what a managed runtime buys you).
- [ ] 15.4 💡 🔭 **Langfuse in production.** Langfuse Cloud, EU region, stays: no operations, data in Ireland, compliance documented by the vendor. Review it with legal as a sub-processor (data processing agreement, data regions, retention) and choose the plan from the 14.11 estimate. Update the ADR from 5.2.
- [ ] 15.5 **Identity.** An IAM role for the task or function with the least-privilege policy from Module 13. No credentials in the image. Langfuse keys from Secrets Manager.
- [ ] 15.6 **The server.** `axum` with health checks (`/healthz`, `/readyz`), graceful shutdown (finish in-flight requests, cancel background tasks sensibly, **flush the OTel exporter**), a per-request timeout, body size limits, per-user rate limiting, and CORS if needed.
- [ ] 15.7 **State.** SQLite is not enough for a multi-instance production: managed PostgreSQL (RDS) with `pgvector` for the knowledge base. Versioned migrations (`sqlx migrate`).
- [ ] 15.8 💡 🔭 **Gradual rollout.** For the app: two versions in parallel (blue/green, or a canary on 5% of traffic). For prompts: the active version is the `production` label in Langfuse, so a prompt rollback is a label move with no deployment; a `canary` label read by 5% of requests lets you try a new prompt on real traffic, with online scores (14.6) filtered by prompt version telling you whether to promote it. Keep the prompt cache short enough (5–10 minutes) for rollback to be fast.
- [ ] 15.9 💡 **Cost controls.** Budgets and alerts are already active (3.8, 14.4); add an emergency kill switch that degrades the service (escalation and fixed replies only) if hourly cost crosses a threshold.
- [ ] 15.10 🧪 **Load testing.** With `k6` or `oha`, 20–50 simulated users sending synthetic messages. Watch for: Bedrock throttling (the quotas from 3.7), p95 per step in Langfuse, router behaviour under stress, cost per minute, trace ingestion lag. Tune quotas, timeouts, the circuit breaker and the exporter's batch size.
- [ ] 15.11 Deployment pipeline: build on tag, push to ECR, deploy to ECS with manual approval for production; the Module 12 nightly must be green before promotion.
- [ ] 15.12 Operational runbook: how to roll back the app and a prompt, how to switch models in an emergency (the router override), how to turn off content tracing, who to call.

**Done when** the app answers in production with streaming, traces arrive in Langfuse with `environment = prod`, alerts are active, and you have rolled back a prompt at least once by moving a label.

**Self-check.** Why must the active prompt version be a label rather than code? What must be reviewed before Langfuse Cloud receives production traces?

---

## Module 16 · Optional deep dives

To explore once the core is solid. Each one is a small experiment with a card in the Work Log.

- [ ] 16.1 **Agent frameworks compared.** Reimplement the Module 7 agent twice, and compare each version with your harness on lines of code, control over the loop, testability, and observability in Langfuse:
  - with **[Rig](https://github.com/0xPlaygrounds/rig)** ([`rig-core`](https://docs.rs/rig-core) + [`rig-bedrock`](https://docs.rs/rig-bedrock)), the most mature Rust framework, with typed tools, RAG, MCP and OpenTelemetry attributes built in: it replaces your loop;
  - with **[Temporal](https://temporal.io)** and its Rust SDK ([`temporalio-sdk`](https://docs.rs/temporalio-sdk), 1.0 since September 2026), which is not an agent framework but a *durable execution* engine: it keeps your loop and makes it survive failures. Add one more dimension to the comparison: kill the process in the middle of a loop and see what happens to the conversation.

  💡 **How durable execution works.** The code is split in two. A *workflow* holds the logic, here the agent loop, and must be deterministic: no network, no clock, no randomness. Everything that touches the outside world, each model call and each tool call, is an *activity*, with its own timeout and retry policy. The Temporal server records every step in the workflow's *event history*; if the worker process dies, another worker replays the history, gets the recorded results of the activities already completed without running them again, and continues from where the loop stopped. A workflow can also wait for days for an external *signal* at no cost, which fits the human approval of 16.13. For this experiment run the local development server (`temporal server start-dev`). ⚠️ The history stores the inputs and outputs of every activity, prompts and replies included: in production it is one more place holding conversation content, so Temporal Cloud would need an EU region like everything else.

  To see what a "complete" framework does, also look at the concepts in LangChain 1.x (Python: `create_agent`, middleware, LangGraph; start from [What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1) and [Agent middleware](https://www.langchain.com/blog/agent-middleware)) without switching stacks: your harness **is** an agent with hand-written middleware. ADR: "framework versus hand-written harness".
- [ ] 16.2 **Multi-agent (orchestrator-workers).** An orchestrator delegates sub-tasks to workers using a cheap model and isolated context, then synthesizes. In Langfuse the trace shows the whole tree. Compare against the single-model pipeline on a knowledge base research task with several questions.
- [ ] 16.3 **Tool search and on-demand loading.** With dozens of tools the list itself costs: techniques for exposing only the tools relevant to a turn (a local index, short descriptions with the full schema on request).
- [ ] 16.4 **Memory as a tool.** Give the model a `memory_read/write` tool over a directory or table and let it decide what to remember; compare with the structured profile from Module 8.
- [ ] 16.5 **Langfuse in more depth.** Composed prompts (a prompt that includes others, to reuse shared rules between `GENERATION` and `GUARDRAILS`); webhooks on prompt changes to trigger the nightly when someone moves `production`; **session-level** scores (the quality of a whole conversation rather than a single message); multimodal traces; batch export to S3 and offline analysis with DuckDB.
- [ ] 16.6 **Bedrock AgentCore in depth.** Runtime, Gateway (tools over MCP with authentication), Memory, Policy with Guardrails, Observability. How much of your code would it replace? Can its traces coexist with Langfuse?
- [ ] 16.7 **Claude Platform on AWS.** Run the same pipeline on Anthropic's managed access inside AWS (parity with the Anthropic API: server-side compaction, context editing, batch). An honest comparison with Bedrock on features, price and data residency.
- [ ] 16.8 **Batch inference.** For `PROFILE_UPDATE` and for evaluations, Bedrock's batch mode is cheaper but asynchronous: when is it worth it?
- [ ] 16.9 **Fine-tuning and distillation versus prompting.** When a small fine-tuned classifier beats a prompt on a large model (`SAFETY` is a candidate). Bedrock offers customization for some models; weigh cost and maintenance. Annotated Langfuse datasets are the starting point for the training set.
- [ ] 16.10 **Migrating to a new model.** Checklist: capabilities (contract tests), full experiments compared in Langfuse against the last good run, **shadow routing** (through the router's override from 11.5, a sample of real requests also goes to the candidate model in the background, tagged `shadow`, without touching the reply, and online scores compare the two), a prompt audit for cruft written for older models, recalibrating `effort`, costs, a production canary with online scores, a rollback ready.
- [ ] 16.11 **Cost optimization as a process.** The order of the levers: caching, input token hygiene (context, tool results), output hygiene, batch, then `effort`, then changing model. Always measure cost per **completed task**, which in Langfuse is cost per trace, not per generation.
- [ ] 16.12 **Multimodal and voice.** Image input (Converse supports image and document blocks), transcription and speech synthesis as pipeline steps.
- [ ] 16.13 **Human approval for dangerous tools.** Add to the agent loop from Module 7 a tool that changes something outside the conversation, for example booking an appointment in a fake calendar, and gate it: the loop suspends, returns a state such as `AwaitingApproval` with the pending call, and resumes only after the user says yes. In Langfuse the trace shows the pause.

---

## Appendix A · Glossary (fill in with your own words)

| Term | Your definition |
|---|---|
| Token | |
| Context window | |
| System prompt | |
| Tool use / function calling | |
| Harness | |
| Agent versus workflow | |
| Structured output | |
| Streaming / time to first token | |
| Prompt caching / stable prefix | |
| Adaptive thinking / effort | |
| Inference profile (cross-region) | |
| Converse API | |
| Guardrail (LLM versus managed) | |
| RAG / embedding / recall@k | |
| Trace / observation / generation | |
| Session / user (Langfuse) | |
| Score (numeric, categorical, boolean) | |
| Prompt version / label | |
| Dataset / dataset run / evaluator / annotation queue | |
| LLM-as-a-judge | |
| Confidence interval / evaluation noise | |
| Router / fallback / circuit breaker | |
| OpenTelemetry / OTLP / GenAI semantic conventions | |
| MCP | |
| ADR | |

---

## Appendix B · Resources

**Concepts and prompting (Anthropic)**
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Claude documentation](https://platform.claude.com/docs/en/home): [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview), [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview), and the prompting notes for [Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5) and [Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5)

**Amazon Bedrock**
- [User guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- [Model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
- [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) and [ConverseStream API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)
- [Structured output](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html)
- [Claude parameters on Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- [Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [Intelligent Prompt Routing](https://aws.amazon.com/bedrock/intelligent-prompt-routing/)
- [AgentCore](https://aws.amazon.com/bedrock/agentcore/)
- [Pricing](https://aws.amazon.com/bedrock/pricing/)

**Langfuse**
- [Documentation](https://langfuse.com/docs)
- [Observability data model](https://langfuse.com/docs/observability/data-model)
- [OpenTelemetry integration](https://langfuse.com/integrations/native/opentelemetry) (endpoint, attributes)
- Prompt management: [concepts, versions and labels](https://langfuse.com/docs/prompt-management/data-model), [caching](https://langfuse.com/docs/prompt-management/features/caching), [guaranteed availability](https://langfuse.com/docs/prompt-management/features/guaranteed-availability), [linking to traces](https://langfuse.com/docs/prompt-management/features/link-to-traces)
- Evaluation: [overview](https://langfuse.com/docs/evaluation/overview), [concepts](https://langfuse.com/docs/evaluation/core-concepts), [LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge), [annotation queues](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues)
- Dashboards and alerts: [custom dashboards](https://langfuse.com/docs/metrics/features/custom-dashboards), [alerts](https://langfuse.com/docs/observability/features/alerts), [Metrics API](https://langfuse.com/docs/metrics/features/metrics-api)
- [Public API](https://langfuse.com/docs/api-and-data-platform/features/public-api)
- [Data regions](https://langfuse.com/security/data-regions) and [EU data residency and GDPR](https://langfuse.com/resources/engineering/langfuse-eu-data-residency-gdpr)
- [Changelog](https://langfuse.com/changelog) (features move fast)
- Rust crates from the genai-rs organization: [`opentelemetry-langfuse`](https://github.com/genai-rs/opentelemetry-langfuse), [`langfuse-ergonomic`](https://github.com/genai-rs/langfuse-ergonomic), [`langfuse-client-base`](https://github.com/genai-rs/langfuse-client-base)

**Rust**
- [Bedrock Runtime examples for Rust](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_bedrock-runtime_code_examples.html)
- [Testing with the SDK](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/testing.html) (mocks and replay)
- [`aws-sdk-bedrockruntime` on docs.rs](https://docs.rs/aws-sdk-bedrockruntime)
- [Official MCP SDK (`rmcp`)](https://github.com/modelcontextprotocol/rust-sdk)
- [Rig](https://github.com/0xPlaygrounds/rig) and the [`rig-bedrock`](https://docs.rs/rig-bedrock) crate (optional comparison)
- [Temporal documentation](https://docs.temporal.io) and the [`temporalio-sdk`](https://docs.rs/temporalio-sdk) crate (optional comparison)
- Crate docs: [`schemars`](https://docs.rs/schemars), [`insta`](https://docs.rs/insta), [`proptest`](https://docs.rs/proptest), [`wiremock`](https://docs.rs/wiremock), [`aws-smithy-mocks`](https://docs.rs/aws-smithy-mocks), [`tracing-opentelemetry`](https://docs.rs/tracing-opentelemetry), [`opentelemetry-otlp`](https://docs.rs/opentelemetry-otlp)

**Observability and security**
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai), now maintained in their own repository
- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
- [Datadog Agent Observability](https://docs.datadoghq.com/llm_observability/) (for correlation with infrastructure)
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

---

## Appendix C · ADR template

File: `docs/adr/NNNN-short-title.md`. The full template lives in `docs/adr/0000-template.md`:

```markdown
# NNNN · Title

- Date: YYYY-MM-DD
- Status: proposed | accepted | superseded by NNNN

## Context
What the problem is, and what the constraints are (cost, latency, privacy, time).

## Options considered
1. …  2. …  3. …

## Decision
What we choose and why, with numbers where they exist (link to the Langfuse dataset run
and to the experiment card).

## Consequences
What becomes easier, what becomes harder, what must be revisited and when.
```

---

## Appendix D · Experiment card

Fill one in the Work Log for every 🧪.

```markdown
### EXP-NNN · Title · YYYY-MM-DD
- Question: …
- Variable changed (exactly one): …
- Configuration: model, prompt@version, effort, dataset@version, repetitions
- Langfuse dataset run: <link>
- Results: quality (with CI), total cost, p50/p95 latency, infrastructure errors
- Conclusion: …
- Next step: …
```

---

## Appendix E · Work log

Append one entry per working session. A few lines: what you did, what you learned, what is still unclear.

```markdown
### YYYY-MM-DD
- Did: …
- Learned: …
- Open questions: …
```

### 2026-09-24
- Did: chose `eu-west-1` as the home region (3.4) and added a dedicated `bedrock-playground` AWS profile, so the shared `sandbox` profile stays on `eu-south-1`. Submitted the Anthropic use case form (3.5). Created `.env` with Haiku 4.5 as the fast model and Sonnet 5 as the main one (3.6). Wrote down the `eu.*` quotas (3.7): Haiku 4.5 has 50 RPM and 5M TPM, Sonnet 5 has 0 TPM, so I requested 6M. Created the `bedrock-playground` budget (3.8): 150 USD a month on the Amazon Bedrock service, email alerts at 50% and 80%.
- Learned: the company SCP denies Bedrock in `eu-central-1`; Milan lists the same Claude 5 `eu.*` profiles as Ireland, but not the Llama and Mistral ones needed in Module 11. `aws bedrock get-use-case-for-model-access` and `get-foundation-model-availability` show model access from the CLI. Fable 5 has only a `global.*` profile, so this project cannot use it. The account's applied quotas are below the AWS defaults (50 RPM instead of 10,000), and a quota of 0 makes an `ACTIVE` model unusable.
- Open questions: `eu.*` profiles also route to `eu-central-1`; check in 3.10 whether the SCP blocks routed calls. The Sonnet 5 quota request is pending: if it is not granted before Module 4, switch `BEDROCK_MODEL_MAIN` to Sonnet 4.6 (50 RPM, 6M TPM). After 3.10, check in Cost Explorer whether Claude costs show up under Amazon Bedrock or as an AWS Marketplace item: in the second case the budget filter misses them.

### 2026-09-25
- Did: enabled text invocation logging (3.9) to `/bedrock-playground/model-invocations`, with 14-day retention. Tried the first CLI call (3.10): denied at first, working after the organization admins changed the SCP. Worked out the cost of that call by hand (3.11). Read about Claude Platform on AWS (3.12). Implemented `complete` over Converse (4.3): the first call from Rust answered.
- Learned: a role that the console has just created can fail validation until IAM propagates it; saving again with the existing role works. The SCP also applies in the region where an `eu.*` profile routes the call: it denies `eu-north-1`, `eu-west-3` and `eu-central-1`, and `eu-south-2` is not enabled, so every Haiku call failed (6 of 6, all routed to `eu-north-1`). First measurement on Haiku 4.5: 15 input and 5 output tokens, `end_turn`, latency 641–789 ms over three identical calls, the first one the slowest. The `inferenceRegion` field of the invocation log shows where a call actually ran: `eu-north-1` for all of them. Bedrock prices in `eu-west-1` for `eu.*` profiles, in USD per million tokens before tax, from the AWS Price List API: Haiku 4.5 costs 1.10 input, 5.50 output, 1.375 cache write (5 minutes), 2.20 cache write (1 hour) and 0.11 cache read; Sonnet 5 costs 2.20, 11.00, 2.75, 4.40 and 0.22. EU prices are 10% above global ones. The 3.10 call cost 15 × 1.10 + 5 × 5.50 per million tokens, about 0.000044 USD. Claude is billed as an AWS Marketplace product ("Claude Haiku 4.5 (Amazon Bedrock Edition)", usage type `EU-MP:…`), so the budget filter on Amazon Bedrock probably misses it. Claude Platform on AWS runs on Anthropic's infrastructure with only global and US inference geographies: today Bedrock is the only option that offers several LLMs with EU data residency, which is why the project stays on it. Sonnet 5 is not available for this account: the quota case was closed without an increase and calls fail with `AccessDeniedException`, so `BEDROCK_MODEL_MAIN` is now Sonnet 4.6, which answers.
- Open questions: why Sonnet 5 is not available, in AWS's answer to support case 179026551500718. The budget filter check is still open.

### 2026-09-26
- Did: mapped Bedrock errors to `LlmError` and wrote out the SDK retry settings (4.4), implemented streaming over ConverseStream (4.5), wired `LlmRequest.extra` into `additionalModelRequestFields` (4.6), verified prompt caching (4.7). Wrote ADR 0003 on the provider-neutral `LlmClient`.
- Learned: real Bedrock errors map as intended: Sonnet 5 gives `AccessDenied`, an invented model id gives `ValidationError`, and an input of 260,024 tokens on Haiku gives `ContextTooLong` ("prompt is too long: 260024 tokens > 200000 maximum"), but only after 14.8 s, so the context size is worth checking before sending. Expired SSO credentials reach Rust as a dispatch failure, mapped to `Other`. A stream delivers text in coarse pieces (32 tokens in 5 deltas), and `Stop` arrives before `Metadata`. The first call of a process waited 1,321 ms for its first token against 546 ms for the next one: credentials and the connection are set up once. Haiku 4.5 rejects `effort` ("This model does not support the effort parameter"). Prompt caching, with three identical calls per case and the `cachePoint` at the end of a synthetic system prompt: on Haiku a 6,631-token prefix is written on the first call and read on the next two (about 0.0091 USD for the first call, 0.0007 for the others, against 0.0073 without the cache); a 2,354-token prefix is silently not cached on Haiku but is cached on Sonnet 4.6, so the minimum depends on the model; a timestamp at the top of the prompt rewrites the cache on every call and never reads it, costing about 25% more than no cache at all. Latency barely changed at this size.
- Open questions: why Sonnet 5 is not available (support case 179026551500718). The budget filter check is still open.

### EXP-001 · Effort on Sonnet 4.6 · 2026-09-26
- Question: what does `effort` change on a short reasoning question with a required format?
- Variable changed (exactly one): `output_config.effort`, `low` against `high`.
- Configuration: `eu.anthropic.claude-sonnet-4-6`, adaptive thinking, `max_tokens` 4000, no system prompt, the question "A clinic has 3 doctors. Each sees 4 patients per hour for 6 hours, and 10% of the appointments are cancelled. How many visits take place? Reply with the number only.", 3 repetitions per level.
- Langfuse dataset run: none yet (Langfuse arrives in Module 5).
- Results: `low` answered 65, 64.8 and 64 with 48–75 output tokens in 1.0–1.4 s; `high` answered 65 three times but added the steps against the instruction, with 350–361 output tokens in 5.1–5.6 s. The correct value is 64.8. About 0.02 USD for the six calls, at 3.30 and 16.50 USD per million input and output tokens. No infrastructure errors.
- Conclusion: `high` costs about 6 times the tokens and 5 times the latency, is more consistent in its value, and follows the format less; `low` is cheap but gives a different answer on each repetition. Three repetitions show the direction, not a measurement with a confidence interval.
- Next step: repeat on a dataset once Langfuse experiments are available (Module 11), measuring format compliance as well as correctness.

### 2026-09-27
- Did: implemented `app hello` (4.8) with a hardcoded EU price table, a `just run` recipe, and a synthetic system prompt in `crates/app/fixtures/clinic-guidelines.md` to test caching. The binary refuses a non-EU region or model id. Added `--stream` to `app hello` (4.9), and fixed the `run` recipe, which split a quoted argument such as `--message "…"` into separate words. Added one `llm_call` span per call with tracing (4.10), which closes Module 4.
- Learned: the cache prices of Sonnet 4.6 in the AWS Price List match the standard ratios: 4.125 USD per million tokens to write (6.60 for one hour) and 0.33 to read, against 3.30 for input. With `RUST_LOG=info`, `aws_config` prints the whole credential chain on every run, so the default is now `info,aws_config=warn`; an error shows in the log with its kind first (`error=invalid request: ValidationException: …`).
- Open questions: why Sonnet 5 is not available (support case 179026551500718). The budget filter check is still open.

### EXP-002 · Latency variance and cache reads with `app hello` · 2026-09-27
- Question: how much does latency vary between identical calls, and is the cache read from the second call on?
- Variable changed (exactly one): none within a scenario, where the same call is repeated; the three scenarios differ by model or by system prompt.
- Configuration: `just run hello` with the default message "Answer with a single word: ready?" and `max_tokens` 50. Scenario 1: Haiku 4.5, no system prompt, 5 calls. Scenario 2: Sonnet 4.6 with `--system crates/app/fixtures/clinic-guidelines.md`, 5 calls. Scenario 3: Haiku 4.5 with the same system prompt, 3 calls.
- Langfuse dataset run: none yet (Langfuse arrives in Module 5).
- Results: scenario 1, Bedrock latency 509–660 ms (mean 560 ms); the first call took 1,282 ms end to end against 557–705 ms for the others; no cache, since 15 input tokens are far below the minimum. Scenario 2, the first call wrote 2,184 tokens to the cache and calls 2 to 5 read them, at 0.0009 USD per call instead of 0.0092; Bedrock latency 698–1,341 ms, wider than Haiku's. Scenario 3, 2,196 input tokens at full price on every call and no cache: the prompt is above Sonnet 4.6's minimum and below Haiku 4.5's. About 0.03 USD in total. No infrastructure errors.
- Conclusion: identical calls vary by about ±15% on Haiku, and the first call of a process pays about 700 ms of setup that Bedrock does not see. The cache works from the second call when the prefix is long enough for the model, and at this size it cuts cost, not latency. Aside, one message only: with a system prompt saying to answer in the person's language, Haiku answered "Ready." and Sonnet 4.6 "Pronto.".
- Next step: measure time to first token with streaming (4.9).

### EXP-003 · Streaming against a full reply · 2026-09-27
- Question: what does streaming change in the time to the first text and in the total time?
- Variable changed (exactly one): `--stream`.
- Configuration: `just run hello` on Haiku 4.5, no system prompt. Short reply: "Write three short sentences about why a clinic should publish its opening hours online.", `max_tokens` 300, 5 calls per mode. Long reply: "Write about 250 words on how a fictional clinic could make its booking process easier for patients.", `max_tokens` 800, 3 calls per mode.
- Langfuse dataset run: none yet (Langfuse arrives in Module 5).
- Results: short reply (68–116 output tokens), total time mean 1,498 ms without streaming and 1,824 ms with it; first token usually after 476–601 ms. Long reply (about 355 output tokens), total time mean 4,330 ms without streaming and 4,401 ms with it; first token usually after 560–640 ms. The first call of each run had its first token at about 1.2 s (client setup, as in EXP-002), and one call at 1.8 s. Output lengths vary between calls, so total times compare only roughly. Less than 0.01 USD. No infrastructure errors.
- Conclusion: streaming does not shorten the reply, since the total time stays within the noise, but the first text appears after about 0.6 s instead of 4.3 s on a long reply, seven times sooner, and the gain grows with the length of the reply.
- Next step: record these timings as spans with `tracing` (4.10), so that Module 5 shows them in Langfuse.
