# Building an agentic application in Rust on Amazon Bedrock, observed with Langfuse

> A step-by-step learning path, written for someone starting from a rough idea of what an LLM harness is.
> Every step is a checkbox: tick `[x]` when it is done. State of the art: September 2026.

---

## 0. How to use this tutorial

### 0.1 Structure

- The modules go in order. Modules **1–9** are the core (concepts, Bedrock, Langfuse, harness, prompts, context, pipeline). Modules **10–15** are the production half (router, evaluation, testing, security, observability, deployment). Module **16** collects optional deep dives.
- Every module has the same sections: **Goal**, **Why it matters**, **Steps** (the checkboxes), **Done when** (an objective completion criterion), **Self-check** (what you must be able to explain out loud), **Going deeper**.
- Legend: ⏱ time estimate · 🧪 experiment to record in the Work Log (Appendix E) · ⚠️ watch out · 💡 key concept · 📚 recommended reading · 🔭 Langfuse-specific step.
- Every non-trivial decision becomes an ADR (Architecture Decision Record, Appendix C). Every experiment with numbers goes in the Work Log. These two habits are worth more than any tool.

### 0.2 Progress

- [ ] Module 1 · Foundations: LLMs, harnesses, agents
- [ ] Module 2 · Environment and Rust repository setup
- [ ] Module 3 · Enabling Amazon Bedrock
- [ ] Module 4 · First Bedrock call from Rust
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
| LLM engineering platform | **Langfuse**, used across all four of its areas: tracing, prompt management, evaluation (datasets, experiments, LLM-as-a-judge, annotations), dashboards and alerts | It is open source, has an EU cloud region and can be self-hosted; it integrates from Rust over OpenTelemetry and its public API. See Module 5. |
| Agent frameworks | **None** in the core: the harness is written by hand. Rig (the most mature Rust framework) is an optional comparison in Module 16 | Writing the loop once is the fastest way to understand what frameworks do for you. |
| Persistence | SQLite locally (`rusqlite` or `sqlx`), PostgreSQL with `pgvector` when vector search is needed | Easy to start, realistic for production. |

---

## Module 1 · Foundations: LLMs, harnesses, agents

⏱ 3–4 hours · reading and notes only, no code.

**Goal.** Build a precise vocabulary before writing a line of code.

**Why it matters.** Seventy percent of the mistakes in agentic applications come from a wrong mental model of what an LLM does and does not do: you write a prompt that "works" without knowing why, and then you cannot debug it.

### Steps

- [ ] 1.1 💡 **What an LLM is, for the person integrating it.** A function `(token sequence) → (distribution over the next token)`, sampled in a loop. It has no memory between calls: everything it "knows" about the conversation is what you send it every time. Write that sentence in your own words in the Work Log.
- [ ] 1.2 💡 **Tokens, context window, cost.** Tokens are the unit of measurement for everything: cost (price per million input and output tokens), limits (the context window), latency (output tokens are paid for in time). Italian text runs at roughly one token per 3–4 characters. The context window of Claude 5 models on Bedrock is 1M tokens, but "it fits" is not "it works well": quality degrades when you fill the window with noise.
- [ ] 1.3 💡 **Message roles.** `system` (operator instructions: who you are, what you do, in what format you answer), `user`, `assistant`. The system prompt is the most powerful lever you have.
- [ ] 1.4 💡 **Inference parameters: how the model picks tokens, and how much it thinks.** These are the knobs you send with every request. They do not change what the model knows; they change how it produces text. They come in two generations, and which ones a model accepts depends on the model.
  - **Sampling knobs.** At every step the model assigns a probability to each candidate token, and one of them must be picked. After "The sky is", for example: *blue* 60%, *clear* 25%, *cloudy* 10%, everything else 5%. `temperature` sets how much risk to take: low means the most likely token almost always wins (predictable, conservative answers); high flattens the distribution, so *cloudy* comes out more often (more varied answers, and more wrong ones). `top_p` cuts the tail: at 0.9 only the most likely tokens that together reach 90% stay in play, which keeps absurd tokens out. Tune one of the two, not both.
  - **Zero is not deterministic.** Even at `temperature = 0` the probabilities wobble very slightly from one call to the next, because of how GPUs compute your request batched together with other people's. When two tokens are nearly tied, that wobble is enough to swap them, and from that token on the whole text takes a different path.
  - **`max_tokens` is a ceiling, not a target.** The model does not know it and does not try to fill it. If it reaches it, the reply stops mid-sentence with `stop_reason = max_tokens`, and a JSON answer cut that way is unusable. You only pay for the tokens actually generated, so a generous ceiling costs nothing. `stop_sequences` stops generation as soon as a given string appears; with structured output you rarely need it.
  - **Thinking and effort.** Recent Claude models can write internal reasoning before the final answer. You do not see it, but you pay for it as output tokens, and it counts toward `max_tokens`: a low ceiling can run out before the answer arrives. It is *adaptive* because the model decides whether and how much to think, based on how hard the request is. You steer it with **effort**, from `low` (little reasoning, fewer tokens, faster and cheaper) to `max` (more reasoning, slower and more expensive, better on hard problems). The default is `high`.
  - **Which model accepts what.** Claude 5 models (Opus, Sonnet, Fable) reject `temperature` and `top_p` with an error: effort is your only lever. Claude Haiku 4.5 is the other way round: it accepts `temperature` and `top_p`, rejects effort, and thinks only when given a fixed token budget. Non-Anthropic models (Nova, Llama, Mistral) accept the sampling knobs and have their own reasoning settings, if any. So the harness must never send the same parameters to every model (4.6, 10.1).
- [ ] 1.5 💡 **Tool calling (function calling).** The model executes nothing: it emits a `tool_use` block with a name and JSON arguments, you run the function and send back a `tool_result` block. The loop `call → tool_use → execute → tool_result → call…` **is the harness**. An "agent" is a harness in which the model decides which tools to use and when to stop.
- [ ] 1.6 💡 **Structured output.** Asking the model to answer with JSON that conforms to a schema. There are two ways: force a tool whose schema is the one you want, or use the native structured output feature (on Bedrock: `outputConfig.textFormat` with `json_schema`). You need it everywhere in the guiding project.
- [ ] 1.7 💡 **Streaming.** Receiving tokens as they are produced instead of all at the end. It changes perceived latency (time to first token) and complicates the code (partial events, tool use arriving in pieces).
- [ ] 1.8 📚 **Workflows versus agents.** Read [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic). Learn the distinction: a **workflow** is where your code decides the flow and the LLM handles individual steps; an **agent** is where the LLM decides the flow. The guiding project is a workflow with agentic parts, which is the right shape for most products.
- [ ] 1.9 📚 **The five composition patterns.** Prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer, all described in [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents). For each one write a line in the Work Log: "in the guiding project I would use this for…". (Hint: `SAFETY` is routing, `GUARDRAILS` is evaluator-optimizer.)
- [ ] 1.10 💡 **When you do NOT need an agent.** Four questions: is the task multi-step and hard to specify up front? Does the value justify the extra cost and latency? Is the model capable at this kind of task? Are errors recoverable? If any answer is no, stay with a single call or a workflow.
- [ ] 1.11 💡 **Non-determinism.** Even at `temperature = 0`, two identical calls can produce different output. This changes everything about testing (Module 12): you do not test "the output is X", you test "the output satisfies properties P, over N repetitions, above a threshold".
- [ ] 1.12 💡 **Observing an LLM system.** Three words we will use constantly: **trace** (everything that happens for one request), **observation** (a piece of a trace: a generic span, a *generation* meaning a model call, an event), **score** (a number or a label attached to a trace or an observation: quality, feedback, guardrail outcome). This is the Langfuse data model, and roughly the industry's.
- [ ] 1.13 Fill in the glossary (Appendix A) in your own words. If a definition does not come to you, you have not understood the concept yet.

**Done when** you have written definitions in the Work Log for: token, context window, system prompt, tool use, harness, agent, workflow, structured output, streaming, effort, trace, observation, score.

**Self-check.** Explain out loud, in two minutes, why "the LLM remembers the conversation" is false, and what your code actually does to create that illusion.

**Going deeper.** [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) (Anthropic Engineering): we return to both in Modules 7 and 8.

---

## Module 2 · Environment and Rust repository setup

⏱ 2–3 hours.

**Goal.** A Cargo workspace with the final structure, minimal CI, and lints and tests running on an empty tree.

**Why it matters.** Separating "what talks to the provider", "what decides the flow" and "what observes" from day one is what makes mocks, tests, the router and tracing possible later.

### Steps

- [ ] 2.1 Check the toolchain. Rust is pinned in `.tool-versions` (asdf); verify with `rustc --version`. Install `cargo install cargo-nextest cargo-watch cargo-lambda just`. Consider `cargo-insta` (snapshot tests) and `cargo-deny` (dependency audit).
- [ ] 2.2 Create the workspace. Suggested layout, one crate per responsibility so the boundaries stay sharp:

  ```text
  agentic-app-playground/
  ├── Cargo.toml                 # [workspace]
  ├── justfile                   # recurring commands (test, eval, run)
  ├── crates/
  │   ├── llm-core/              # neutral types: Message, ContentBlock, ToolSpec, LlmRequest/Response, LlmClient trait
  │   ├── llm-bedrock/           # LlmClient implementation over aws-sdk-bedrockruntime (Converse)
  │   ├── observability/         # tracing + OpenTelemetry init, Langfuse exporter, attribute helpers, Langfuse API client
  │   ├── prompts/               # prompt loading: from Langfuse (with cache) falling back to files in the repo
  │   ├── harness/               # agent loop, tool registry, context management, pipeline steps
  │   ├── router/                # ModelRegistry, ModelRouter, fallback, circuit breaker
  │   ├── evals/                 # evaluation runner: datasets, graders, Langfuse experiments, reports
  │   └── app/                   # binary: CLI (clap) and HTTP server (axum) with SSE streaming
  ├── prompts/                   # prompt snapshots (fallback and PR diffs), one per step
  ├── datasets/                  # test cases and golden sets (JSONL), synthetic data only
  ├── docs/adr/                  # Architecture Decision Records
  └── TUTORIAL.md
  ```

- [ ] 2.3 Baseline workspace dependencies: `tokio` (full), `serde`, `serde_json`, `schemars` (generates JSON Schema from structs, which you will use for tools and structured output), `thiserror`, `anyhow`, `tracing`, `tracing-subscriber` (env-filter, json), `dotenvy`, `clap`, `async-trait`, `futures`. Take the versions from crates.io on the day you install: do not trust numbers read in a tutorial.
- [ ] 2.4 Fill in `.env` from `.env.example` (`.env` is never committed) with `AWS_PROFILE`, `AWS_REGION`, `BEDROCK_MODEL_FAST`, `BEDROCK_MODEL_MAIN` and the Langfuse placeholders, which you complete in Module 5.
- [ ] 2.5 Lints and formatting: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`. A `justfile` with `just check`, `just test`, `just eval`, `just run`.
- [ ] 2.6 CI on GitHub Actions: a `ci.yml` workflow that on every pull request runs fmt, clippy and `cargo nextest run` **with no credentials** (the tests that talk to Bedrock or Langfuse will be `#[ignore]`, Module 12). A second workflow, `nightly.yml`, empty for now, which will later run contract tests and experiments.
- [ ] 2.7 Write `docs/adr/0001-workspace-layout.md` from the template in `docs/adr/0000-template.md`. It is your first ADR: short is fine, five lines is fine.
- [ ] 2.8 Commit (English, imperative: `Add workspace skeleton and CI`).

**Done when** `just check` and `just test` pass on an empty workspace and CI is green.

**Self-check.** Why must `llm-core` not depend on `aws-sdk-*` or on `observability`? (So you can mock the provider in tests, add a second provider without touching the harness, and swap the observability backend without touching the logic.)

---

## Module 3 · Enabling Amazon Bedrock

⏱ 2–3 hours, much of it waiting on the AWS console.

**Goal.** An AWS account with Bedrock enabled in the EU, least-privilege credentials, models enabled, a budget in place and a successful first call from the CLI.

**Why it matters.** Bedrock's access model is different from an "API key" provider: IAM, regions, model enablement, quotas. Getting this wrong produces cryptic errors (`AccessDeniedException`, `ValidationException` on a model id) that look like bugs in your code.

### Steps

- [ ] 3.1 **Account.** Use a dedicated **sandbox** AWS account (if the company has AWS Organizations, ask for one; otherwise a personal account). MFA on the root user, and never work as root.
- [ ] 3.2 **Identity.** Prefer IAM Identity Center (SSO) with a permission set; otherwise an IAM user with MFA and rotated access keys. To get going, the managed policy `AmazonBedrockFullAccess` is fine; **before Module 15** replace it with a policy granting only `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (these are the actions Converse and ConverseStream use too) on the ARNs of the inference profiles you actually use.
- [ ] 3.3 **AWS CLI.** Already installed (v2.36). Configure a profile: `aws configure sso` or `aws configure --profile bedrock-playground`. Verify with `aws sts get-caller-identity --profile bedrock-playground`.
- [ ] 3.4 **Region.** Pick `eu-west-1` (Ireland) or `eu-central-1` (Frankfurt) as your home region. ⚠️ Milan (`eu-south-1`) has a thinner model catalogue. 💡 With **cross-region inference** (an inference profile prefixed `eu.`) Bedrock routes across EU regions while staying inside the geography: more capacity, and data that does not leave the EU. The `global.` prefix can leave the EU: do not use it in this project.
- [ ] 3.5 **Enabling models.** Console → Bedrock → *Model catalog*. **Anthropic** models require a *use case* form, filled in once per account, with access granted immediately on submission. Also enable **Amazon Nova** (no form) and at least one open-weight model (Llama or Mistral) for Module 11. The AWS Marketplace subscription is created automatically on first invocation if the identity has the right permissions.
- [ ] 3.6 **Find the right ids.** Ids change: do not copy them from tutorials, read them from your account.

  ```bash
  aws bedrock list-foundation-models --region eu-west-1 --by-provider anthropic --query 'modelSummaries[].modelId'
  aws bedrock list-inference-profiles --region eu-west-1 --query 'inferenceProfileSummaries[].inferenceProfileId'
  ```

  Example shapes, to be verified: `eu.anthropic.claude-sonnet-5`, `eu.anthropic.claude-opus-5`, `eu.anthropic.claude-haiku-4-5-20251001-v1:0`. Record a **fast and cheap** model in `.env` (`BEDROCK_MODEL_FAST`, typically Haiku) and a **main** one (`BEDROCK_MODEL_MAIN`, typically Sonnet).
- [ ] 3.7 **Quotas.** Console → Service Quotas → Bedrock: look for *requests per minute* and *tokens per minute* for your chosen models. Write them down: they are the ceiling the router (Module 10) has to respect. Request an increase only if you need one.
- [ ] 3.8 **Budget.** AWS Budgets: a monthly budget (say 30 €) with email alerts at 50% and 80%. Also turn on *Cost allocation tags* with a `project=agentic-playground` tag, which you will use to reconcile the costs Langfuse computes against the bill (Module 14).
- [ ] 3.9 **Invocation logging.** Console → Bedrock → *Settings* → *Model invocation logging* to CloudWatch Logs. Useful in the first days of debugging; once you have Langfuse (Module 5) it becomes redundant. ⚠️ Those logs contain full prompts: in production with personal data this choice must be revisited (Module 13).
- [ ] 3.10 🧪 **First call from the CLI.**

  ```bash
  aws bedrock-runtime converse \
    --region eu-west-1 --profile bedrock-playground \
    --model-id "$BEDROCK_MODEL_FAST" \
    --messages '[{"role":"user","content":[{"text":"Answer with a single word: ready?"}]}]' \
    --inference-config '{"maxTokens":50}'
  ```

  Look at the response: `output.message.content`, `stopReason`, `usage.inputTokens`, `usage.outputTokens`, `metrics.latencyMs`. Record the tokens and the latency in the Work Log: that is your first measurement.
- [ ] 3.11 📚 **Pricing.** Read the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) for your chosen models. Work out by hand what the call you just made cost. Learn the difference between input tokens, output tokens, **cache writes** and **cache reads** (cache reads cost roughly 10% of normal input). You will enter these prices into Langfuse in Module 5.
- [ ] 3.12 💡 **A note on "Claude Platform on AWS".** Since 2026 Anthropic also offers access operated directly by Anthropic inside AWS (SigV4 authentication, Marketplace billing) with feature parity with the Anthropic API and unprefixed model ids. It is an alternative to Bedrock, not the same thing: Bedrock is operated by AWS and has its own multi-vendor catalogue, Guardrails, Knowledge Bases and AgentCore. This tutorial stays on Bedrock because the point is to learn the provider; we come back to it in Module 16.

**Done when** the call in 3.10 answers, the budget is active, and `.env` holds two valid EU-region model ids.

**Self-check.** What happens if you call a model with a `us.*` id from `eu-west-1`? And why is an `eu.*` inference profile preferable to a bare, unprefixed model id?

---

## Module 4 · First Bedrock call from Rust

⏱ 4–6 hours.

**Goal.** An `llm-bedrock` crate implementing the `LlmClient` trait defined in `llm-core`, with a simple call, streaming, error handling, and token and latency measurement.

**Why it matters.** This is where the most important boundary in the architecture is born: the harness will only ever talk to `LlmClient`, never to the SDK.

### Steps

- [ ] 4.1 Define the neutral types in `llm-core`. Do not copy the SDK's types: model only what you need.

  ```rust
  pub enum Role { User, Assistant }
  pub enum ContentBlock { Text(String), ToolUse { id: String, name: String, input: serde_json::Value },
                          ToolResult { tool_use_id: String, content: String, is_error: bool }, CachePoint }
  pub struct Message { pub role: Role, pub content: Vec<ContentBlock> }
  pub struct LlmRequest { pub model: ModelId, pub system: Vec<SystemBlock>, pub messages: Vec<Message>,
                          pub tools: Vec<ToolSpec>, pub max_tokens: u32, pub output_schema: Option<schemars::Schema>,
                          pub extra: serde_json::Value /* model-specific fields, e.g. effort */ }
  pub struct LlmResponse { pub message: Message, pub stop_reason: StopReason, pub usage: Usage, pub latency: Duration }
  pub struct Usage { pub input_tokens: u32, pub output_tokens: u32, pub cache_read_tokens: u32, pub cache_write_tokens: u32 }

  #[async_trait::async_trait]
  pub trait LlmClient: Send + Sync {
      async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError>;
      async fn stream(&self, req: LlmRequest) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError>;
  }
  ```

- [ ] 4.2 In `llm-bedrock` add `aws-config` (feature `behavior-version-latest`) and `aws-sdk-bedrockruntime`. Build the client once, because it is expensive, and share it with `Arc`.

  ```rust
  let cfg = aws_config::defaults(BehaviorVersion::latest()).region(Region::new("eu-west-1")).load().await;
  let client = aws_sdk_bedrockruntime::Client::new(&cfg);
  ```

- [ ] 4.3 Implement `complete` with `client.converse()`: `model_id`, `system(SystemContentBlock::Text(..))`, `messages(..)`, `inference_config(InferenceConfiguration::builder().max_tokens(..))`. Map the response: `output()` → `ConverseOutput::Message`, `stop_reason()`, `usage()`, `metrics().latency_ms()`. 📚 Check [`aws_sdk_bedrockruntime` on docs.rs](https://docs.rs/aws-sdk-bedrockruntime) for the exact names: the SDK's builders change between versions.
- [ ] 4.4 **Errors.** Map the service exceptions to an `LlmError` whose variants matter to the caller: `Throttled` (retryable), `ModelNotReady`/`ServiceUnavailable` (retryable), `ValidationError` (your bug, do not retry), `AccessDenied` (configuration), `ContextTooLong`, `Other`. 💡 The SDK already retries with exponential backoff and jitter: configure it (`RetryConfig`) instead of rewriting it; the router (Module 10) will only add cross-model fallback.
- [ ] 4.5 **Streaming.** Implement `stream` with `converse_stream()`: you receive `ConverseStreamOutput` with `MessageStart`, `ContentBlockStart`, `ContentBlockDelta` (text or fragments of the tool's JSON), `ContentBlockStop`, `MessageStop` and `Metadata` (usage) events. Translate them into your own `LlmEvent`. ⚠️ Tool arguments arrive as a JSON string in pieces: accumulate them and parse only at `ContentBlockStop`.
- [ ] 4.6 **Model-specific parameters.** Converse splits parameters in two. `inferenceConfig` holds the fields every model shares (`maxTokens`, `temperature`, `topP`, `stopSequences`); anything vendor-specific goes in `additional_model_request_fields` (a `Document`). For Claude, `thinking` and `output_config.effort` travel there, among others. Wire `LlmRequest.extra` into that field. ⚠️ Models do not accept the same parameters (see 1.4): Claude 5 rejects `temperature`, Claude Haiku 4.5 rejects `effort`. Start with a small capability table in `llm-bedrock`, keyed by model family, that drops any parameter the target model does not accept, logs a warning, and records the parameters actually sent on the call's span (4.10), so that an experiment never believes it changed a variable that was dropped. In Module 10 this table moves into the `ModelRegistry`.
- [ ] 4.7 **Prompt caching.** Converse supports a `cachePoint` block: it marks the end of the stable prefix (system prompt, tool definitions). Add `ContentBlock::CachePoint` and verify that `usage.cache_read_tokens` becomes greater than zero on the second identical call. If it stays at zero, something in the prefix changes on every call (a timestamp, field ordering).
- [ ] 4.8 🧪 An `app hello` binary that makes a call and prints the reply, stop reason, tokens, latency and estimated cost (a hardcoded price table for now). Repeat the same call five times: record the latency variance and whether the cache is being read.
- [ ] 4.9 🧪 Streaming from the CLI: print tokens as they arrive. Measure **time to first token** and total time. Compare them with the non-streaming call.
- [ ] 4.10 Add `tracing`: one span per call with `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `latency_ms` and `stop_reason` fields. In Module 5 these spans become Langfuse *generations* without touching `llm-bedrock`.
- [ ] 4.11 A second, fake `LlmClient` called `FakeLlmClient`, in `llm-core` behind a `test-util` feature: it returns preconfigured responses. You need it from Module 7 onwards to test the harness without a network.

**Done when** `app hello` and `app hello --stream` work, throttling and validation errors are distinguishable in the logs, and `cache_read_tokens > 0` on the second call.

**Self-check.** Why must tool arguments be parsed only at the end of the block when streaming? Which errors are retryable and which are not?

**Going deeper.** The official [Bedrock Runtime examples for Rust](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_bedrock-runtime_code_examples.html) in the *AWS SDK for Rust Developer Guide*; the [`Converse`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) and [`ConverseStream`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html) API reference.

---

## Module 5 · Enabling Langfuse and seeing the calls

⏱ 4–6 hours.

**Goal.** A live Langfuse project (EU cloud region, or self-hosted locally), the Rust app exporting every call as a *generation* with tokens, cost, model and latency, and you being able to read a trace.

**Why it matters.** From here on, every experiment in this tutorial is read in Langfuse, not in the logs. Seeing the calls while you learn shortens the feedback loop more than any amount of reading. And it is the tool you want to master.

### Steps

- [ ] 5.1 📚 🔭 **The data model.** Read [Observability Data Model](https://langfuse.com/docs/observability/data-model) in the Langfuse docs. Fix these in your mind: **trace** (one end-to-end request, with `input`, `output`, `user_id`, `session_id`, `tags`, `metadata`, `release`, `version`, `environment`), **observation** (span, generation, event, plus the agentic types agent/tool/chain/retriever/evaluator; nestable), **generation** (one model call: model, parameters, usage, cost, linked prompt), **score** (numeric, categorical or boolean; on a trace, an observation or a session), **session** (several traces from the same conversation), **dataset** and **dataset run** (you meet these in Module 11).
- [ ] 5.2 🔭 **Decide where Langfuse runs.** Two routes, both fine for learning:
  - **Langfuse Cloud, EU region** (`https://cloud.langfuse.com`, data in Ireland `eu-west-1`, the same geography as your Bedrock). Zero infrastructure, and the free tier is enough for this tutorial.
  - **Self-hosted locally** with Docker Compose: `web` and `worker` components, PostgreSQL, ClickHouse, Redis/Valkey, S3/MinIO. Useful for understanding the architecture (asynchronous queue-based ingestion, analytics on ClickHouse) and for Module 15, where you evaluate self-hosting on AWS.
  Recommendation: start on EU Cloud and do the local self-host as a Module 15 experiment. Write the choice in an ADR with "data residency" as the rationale.
- [ ] 5.3 🔭 Create an organization and a **project** (`agentic-playground`). Generate an **API key** pair (public and secret) and put them in `.env`. Langfuse keys are per project: different environments (local, dev, prod) can be different projects **or** the same project with an `environment` attribute on traces. For this tutorial use one project plus `environment`.
- [ ] 5.4 🔭 **Models and prices.** Langfuse computes cost from `usage` only if it knows the model. Under *Settings → Models*, add definitions for your Bedrock model ids (matched by a regex on the name, for example `(?i)^eu\.anthropic\.claude-haiku-4-5.*`) with per-token prices for input, output, cache read and cache write, taken from 3.11. Without this step you will see tokens but no cost.
- [ ] 5.5 💡 **How you reach Langfuse from Rust.** Langfuse has no official Rust SDK; it has two doors open to any language: the **OpenTelemetry endpoint** (`/api/public/otel`, OTLP over HTTP, Basic authentication with `public:secret`) and the **public REST API**. Use OTel for tracing and the API for prompts, datasets, scores and experiments. Useful crates: [`opentelemetry-langfuse`](https://github.com/genai-rs/opentelemetry-langfuse) (a builder for an OTLP exporter preconfigured for Langfuse) and [`langfuse-ergonomic`](https://github.com/genai-rs/langfuse-ergonomic) (a public-API client with builders, on top of the OpenAPI-generated [`langfuse-client-base`](https://github.com/genai-rs/langfuse-client-base)).
- [ ] 5.6 In the `observability` crate, initialize the stack: `tracing` → `tracing-opentelemetry` → `opentelemetry_sdk` with an OTLP exporter pointing at Langfuse (`opentelemetry-otlp` with `endpoint = {LANGFUSE_HOST}/api/public/otel` and an `Authorization: Basic base64(public:secret)` header, or the builder from `opentelemetry-langfuse`). Keep the `fmt` layer on stdout for development. Use the batch exporter, and flush explicitly on shutdown, or CLI binaries will lose their last traces.
- [ ] 5.7 🔭 **Mapping spans onto the data model.** Langfuse reads both the OpenTelemetry GenAI conventions and its own `langfuse.*` attributes. The minimum rules:
  - The **root** span becomes the trace: put `langfuse.session.id`, `langfuse.user.id` (already pseudonymized), `langfuse.trace.tags`, `langfuse.environment`, `langfuse.release` (the app version), `langfuse.trace.input` and `langfuse.trace.output` on it.
  - The span of an LLM call becomes a generation, with `langfuse.observation.type = "generation"`, `gen_ai.request.model` (or `langfuse.observation.model.name`), `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and the cache details in `langfuse.observation.usage_details` (a JSON object with `input`, `output`, `cache_read_input_tokens`, `cache_creation_input_tokens`); `langfuse.observation.input` and `.output` carry the content (see 5.10 for privacy).
  - Tool spans: `langfuse.observation.type = "tool"`. Pipeline steps: `"span"` or `"chain"`.
  📚 Check the exact names in the [OpenTelemetry page](https://langfuse.com/integrations/native/opentelemetry) of the Langfuse docs: the attribute list evolves.
- [ ] 5.8 Write a helper in `observability` that the harness uses without knowing anything about Langfuse: `record_generation(span, &LlmRequest, &LlmResponse)`, which sets the 5.7 attributes from your neutral types. Wire it to the span from 4.10.
- [ ] 5.9 🧪 Run `app hello` three times. In Langfuse open *Tracing → Traces*: you should see three traces, each with one generation, showing model, tokens, **cost** and latency. Open a generation and read its input, output and usage. If cost is missing, go back to 5.4. If the trace is missing, check the flush and the credentials.
- [ ] 5.10 💡 🔭 **Content and privacy, first rule.** Decide **now** on a policy for `input`/`output`: full content in `local` and `dev`; in `prod` they pass through a redaction function (Module 13) and through sampling. Implement it as configuration in `observability`, not as scattered `if`s. Langfuse also offers masking at ingestion and per-project retention: note both options for Module 13.
- [ ] 5.11 🧪 Simulate a conversation from the CLI (`app chat` with three messages): each message is a trace, all sharing the same `session.id`. Open *Sessions* in Langfuse: you should see the whole conversation in order. Then *Users*: you should see the synthetic user with cumulative cost.
- [ ] 5.12 🧪 Add your first **score** through the API (`langfuse-ergonomic` or `POST /api/public/scores`): after `app hello`, attach a boolean `smoke_ok = true` score to the trace. This teaches you the mechanism you will use for guardrails (Module 9), user feedback and evaluation (Modules 11 and 14).
- [ ] 5.13 🔭 Take a full tour of the interface and note in the Work Log what each section is for: Tracing (Traces, Sessions, Users, Observations), Prompts, Evaluation (Datasets, Evaluators, Scores, Annotation Queues), Dashboards, Settings (Models, API keys, Members, Retention). It is the map of the next modules.

**Done when** every call from the app appears in Langfuse as a generation with a cost, conversations are grouped into sessions, a score arrives through the API, and the content policy lives in configuration.

**Self-check.** Why do trace attributes (`session.id`, `user.id`, `tags`) belong on the root span and not only on a generation? What is the difference between one Langfuse project per environment and a single project with an `environment` attribute?

**Going deeper.** The [OpenTelemetry](https://langfuse.com/integrations/native/opentelemetry) and [Observability Data Model](https://langfuse.com/docs/observability/data-model) pages in the Langfuse docs; the READMEs of the [`opentelemetry-langfuse`](https://github.com/genai-rs/opentelemetry-langfuse) and [`langfuse-ergonomic`](https://github.com/genai-rs/langfuse-ergonomic) crates; [Self-hosting](https://langfuse.com/self-hosting) for the v3 architecture.

---

## Module 6 · Systematic prompt engineering with Langfuse Prompt Management

⏱ 8–10 hours, spread over several days.

**Goal.** A repeatable method for writing, measuring and versioning prompts, with Langfuse as the prompt registry and repository files as the fallback, applied to the guiding project's `SAFETY` prompt.

**Why it matters.** "The best possible prompt" does not exist in the abstract: what exists is the best trade-off between quality, cost and latency **for one task, one model and one measured dataset**. Without measurement you go in circles. And without a versioned registry you never know which prompt produced which answer.

### Steps

- [ ] 6.1 📚 Read the [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) in the Claude documentation, then [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), and the prompting notes for the Claude 5 models you will use ([Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5), [Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5)): prompts written for older models tend to be over-prescriptive and degrade output on recent ones.
- [ ] 6.2 💡 **Anatomy of a system prompt.** In order: (1) role and operating context, (2) the task's goal, (3) rules and constraints (positive: "do X" works better than "do not do Y"), (4) output format, (5) examples (few-shot) only if they earn their place, (6) variable data **last**. XML tags (`<context>`, `<rules>`, `<examples>`) help the model separate the parts and are the standard way to delimit user input from everything else.
- [ ] 6.3 📚 🔭 **Prompt Management in Langfuse.** Read [Prompt Management Concepts](https://langfuse.com/docs/prompt-management/data-model). Fix these: a prompt has a **name**, immutable **versions** (1, 2, 3…), **labels** pointing at a version (`production` is what is served by default; `staging`, `latest` and custom labels), a **text** or **chat** type, **variables** with `{{name}}` syntax, a `config` (free-form JSON: put the model, effort and `max_tokens` there), tags, and the ability to compose prompts inside prompts. `production` labels can be **protected** so only certain roles move them.
- [ ] 6.4 🔭 Create the `safety-classifier` prompt in Langfuse (chat type: a system message with the rules and a user message with `{{message}}`), with `config = { "model_role": "fast", "effort": "low", "max_tokens": 1024 }`. Assign the `production` label to v1. The `config` holds hints, not guarantees: if the fast model is Claude Haiku 4.5, which rejects effort, the capability table from 4.6 drops that field and the span shows it. The ceiling is generous on purpose: the answer is a small JSON, but a model that thinks before answering spends part of the ceiling on reasoning, and unused ceiling costs nothing.
- [ ] 6.5 🔭 In the `prompts` crate implement `PromptStore::get(name, label)`:
  1. call `GET /api/public/v2/prompts/{name}?label=production` (through `langfuse-ergonomic`),
  2. **cache in memory** with a TTL (60 seconds is the official SDKs' default, see [Caching](https://langfuse.com/docs/prompt-management/features/caching); 5–10 minutes is fine in production),
  3. **fall back** to the `prompts/<name>.md` file in the repository when Langfuse does not answer (this is the [guaranteed availability](https://langfuse.com/docs/prompt-management/features/guaranteed-availability) pattern from the docs),
  4. return an object with `name`, `version`, `template`, `config` and a `render(vars)` method.
  A `just prompts-pull` job downloads snapshots of the `production` versions into `prompts/`, so every prompt change also shows up in a pull request diff and the fallback stays current.
- [ ] 6.6 🔭 **Link prompts and generations.** When the harness makes a call using a prompt, set `langfuse.observation.prompt.name` and `langfuse.observation.prompt.version` on the generation's span. Opening the prompt in Langfuse then shows every generation that used it, with average cost and latency **per version**: this is the basis of every comparison.
- [ ] 6.7 💡 **Prefix stability.** What does not change between calls (instructions, schema, examples) goes first; what does change (user profile, retrieved context, the message) goes after the `cachePoint`. Every byte that changes in the prefix invalidates the cache. With Langfuse variables this translates to: variables only in the final part of the template.
- [ ] 6.8 💡 **Structured output.** Define the output schema as a Rust struct with `serde` and `schemars`, generate the JSON Schema and pass it in `outputConfig.textFormat` (Converse, `json_schema` type). Alternatively, force a tool with that schema. ⚠️ Not every model supports both routes, and Claude Fable 5.1 does not accept forced tool choice: test this per model and record the result in the `ModelRegistry` (Module 10). The schema lives in the code, not in the prompt: the Langfuse prompt only references the schema by name.
- [ ] 6.9 💡 **Thinking and effort.** On models that support it (Claude 5, see 1.4) reasoning is adaptive and you tune its depth with `effort` (`low`, `medium`, `high`, `xhigh`, `max`). The main generation runs on a Claude 5 model: try `medium` and `high` there and measure. For a classification task such as `SAFETY`, `low` is usually enough on a Claude 5 model; on Claude Haiku 4.5 the lever does not exist, and what you tune is the prompt itself and, at most, `temperature`. 💡 Before moving to a bigger model, try the current model at higher effort when it supports effort: it is often cheaper. These values belong in the prompt's `config`, so they are versioned together with the text.
- [ ] 6.10 🧪 **Lab: the `SAFETY` prompt in Italian.** Build `datasets/safety/v1.jsonl` with **30–50 synthetic messages** labelled by hand (`SAFE`, `PSYCH_CRISIS`, `MEDICAL_EMERGENCY`), including the ambiguous cases typical of Italian ("non ce la faccio più", "sono stanca di tutto", "mi scoppia la testa"). Measure accuracy, false negatives (the worst case here), latency and cost on `BEDROCK_MODEL_FAST`. For now the runner is a simple script; in Module 11 it becomes a Langfuse experiment.
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

- [ ] 7.1 💡 Define `Tool` as a trait: `name()`, `description()`, `input_schema()` (generated by `schemars` from an input struct), `async fn call(&self, input: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>`. A `ToolRegistry` holds them by name.
- [ ] 7.2 Translate your `ToolSpec` values into Converse's `ToolConfiguration` (name, description, `inputSchema.json`), both streaming and not.
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
- [ ] 7.4 **Limits.** Maximum iterations, an overall timeout, a token budget, and a list of "dangerous" tools requiring human approval: the loop suspends and returns an `AwaitingApproval` state the caller can resume. The guiding project has no dangerous tool yet, but the mechanism is needed.
- [ ] 7.5 📚 **Designing good tools.** Read [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents). Practical rules: few tools with sharp boundaries; names and descriptions written for the model (they say *when* to use it and *when not*); small, typed inputs; compact outputs (the result lands in the context and is paid for on every later turn); descriptive errors that tell the model how to correct itself.
- [ ] 7.6 🔭 **Nested traces.** The agent's span (`langfuse.observation.type = "agent"`) contains one generation per iteration and one `"tool"` span per tool execution, with input and output. In Langfuse the trace should read as a tree: iteration 1 → tool A, tool B → iteration 2 → final answer. Add `iterations` and `tool_calls` to the trace metadata.
- [ ] 7.7 🧪 Example tools for the guiding project: `search_knowledge_base(query, top_k)` (for now over an in-memory list with text search; it becomes vector search in Module 8), `get_user_profile(user_id)`, `get_current_date()`. Build a demo where the model decides on its own when to search, and read it in Langfuse.
- [ ] 7.8 **Tests without a network.** With `FakeLlmClient`, write tests verifying that: the loop stops at `EndTurn`; a nonexistent tool produces `is_error`; the iteration ceiling fires; two `tool_use` blocks in one message are executed in parallel and returned in a single message.
- [ ] 7.9 💡 **Model Context Protocol (MCP).** This is the standard for exposing tools to a model through a separate server. With the official `rmcp` crate, write a **client** that connects to an MCP server (for example a sample filesystem server, or one of your own exposing the knowledge base) and adapts its tools to your `Tool` trait. The harness then cannot tell local tools from remote ones.
- [ ] 7.10 🧪 Measure in Langfuse how many tokens the tool list alone costs (compare a generation with and without tools). With many tools that cost is paid on every turn, which is why on-demand loading techniques exist (Module 16).
- [ ] 7.11 💡 **Structured output through a tool.** Implement the alternative from 6.8: an `emit_result` tool with the schema you want. Compare reliability and latency against `outputConfig.textFormat`. Record the result per model.

**Done when** the generic loop is tested without a network, a remote MCP tool is usable exactly like a local one, the Langfuse trace shows the agent → generation → tool tree, and you have a demo where the model searches the knowledge base when it needs to.

**Self-check.** What happens if a tool returns 50 KB of text? (The context explodes, cost grows on every turn, quality drops.) What are three ways to avoid it?

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
- [ ] 9.7 Expose the pipeline over HTTP with `axum`: `POST /chat` (full reply) and `POST /chat/stream` (SSE). Fake authentication for now (a header carrying `user_id`). Return the `trace_id` in the response body: you need it to attach user feedback (Module 14).
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

- [ ] 10.1 💡 **ModelRegistry.** A table of `ModelSpec` entries loaded from configuration (TOML): `id`, `provider`, `tier` (small/mid/large), input/output/cache price per Mtok, context window, `supports_tools`, `supports_structured_output`, `supports_streaming`, `accepted_params` (which of `temperature`, `top_p`, `effort` and `thinking` the model takes; it replaces the capability table from 4.6), `region_profile`, quotas (rpm, tpm), measured p50 latency. You **verify capabilities with tests** (Module 12), you do not copy them from the docs. The prices must match the ones entered in Langfuse (5.4): keep them in one file and generate both.
- [ ] 10.2 💡 **Routing strategies**, in increasing order of complexity:
  1. **Static per step**: `SAFETY → small`, `GENERATION → mid/large`, `GUARDRAILS → small`, `PROFILE_UPDATE → small`. This covers 90% of real cases. The prompt's `config` in Langfuse (6.4) names the **role** (`fast`, `main`) and the router translates a role into a model, so changing the model does not mean touching the prompt, and vice versa.
  2. **Rule-based**: on input length, language, presence of tools, structured output requirements, premium users.
  3. **Cascading**: try the cheap model; if its stated confidence is low or the output fails schema validation, escalate to the bigger one. Measure how often it escalates.
  4. **Classifier-driven**: a small model (or a rule) estimates difficulty and picks the tier. Mind the cost of the extra call.
  5. **Managed**: *Amazon Bedrock Intelligent Prompt Routing* picks among models in a family based on the prompt. Try it to see what it does, and compare it with your strategy 3.
- [ ] 10.3 💡 **Fallback and resilience.** A chain per role: primary → secondary (same tier, different model or a different EU region) → degradation (a courtesy reply, a queue). Retry only on retryable errors, with backoff and jitter (the SDK does this for a single call; the router does it **across models**). A circuit breaker per model: after N errors in T seconds, skip to the secondary for a while, then probe again gradually.
- [ ] 10.4 💡 **Constraints to respect.** The request's remaining timeout (falling back makes no sense with 200 ms left); per-model quotas (a local token bucket with `governor` keeps you out of throttling); region (never leave the EU); capability (do not route a request with tools to a model that does not support them).
- [ ] 10.5 ⚠️ **The cache is per model.** Switching models mid-conversation invalidates the prompt cache and can change style and behaviour. The router should prefer **per-session stability**: pin the model at the start of a conversation and change it only on errors, not for instantaneous optimization.
- [ ] 10.6 🔭 **The router in Langfuse.** Every generation carries `router.role`, `router.reason` (`static`, `fallback:throttled`, `escalation:low_confidence`) and `router.attempt` in its metadata; the model actually used is already in the model field. In Langfuse build a **dashboard** with: generation distribution per model, cost per role, fallback counts (filtered on metadata). You will see immediately if a cheap model escalates too often.
- [ ] 10.7 💡 **Controlled experimentation.** Support a per-request `override` (a header or config) to force a model: you need it for the experiments in Module 11. Add *shadow routing*: the request goes to model A and a sample (say 5%) **also** goes to B in the background purely for comparison, without affecting the user's reply; the shadow generation carries a `shadow` tag.
- [ ] 10.8 🧪 Simulate failures with `FakeLlmClient` (a burst of throttling, timeouts, invalid output) and verify: correct fallback, a circuit breaker that opens and closes again, and no request exceeding the configured quota.
- [ ] 10.9 🧪 With real models: compare the static strategy against the cascade over 50 synthetic messages. Total cost, p50/p95 latency, quality (Module 11). Write the ADR with the decision.

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
- [ ] 11.3 🔭 **Datasets.** Source of truth: JSONL files under `datasets/`, versioned in git. A `just datasets-push` command uploads them to Langfuse as datasets (`safety-v1`, `generation-v1`…) through the API. Where the cases come from: written by hand, synthesized with a model **and then reviewed by a person**, or (in production) from real traces added to the dataset with one click in the Langfuse interface. ⚠️ Never use the output of the model under evaluation as ground truth. Balance the classes. For `SAFETY`, include cases in both directions explicitly (must fire / must not fire).
- [ ] 11.4 💡 **Types of grader.** (1) Programmatic: exact match, schema validation, regex, checking that cited knowledge base ids exist. (2) Rubric with an LLM judge: **atomic** properties scored one at a time (`no_diagnosis`, `grounded_in_kb`, `tone_ok`), with structured output, treating the candidate text as data rather than instructions. (3) Pairwise: two answers, which is better (randomize the A/B order). Use a judge from a **different family** than the model under evaluation when you can.
- [ ] 11.5 🔭 **The runner as a dataset run.** `just eval --dataset safety-v1 --model X --prompt safety-classifier@v3 --reps 3` does this: create a run with a descriptive name (`safety-v1 / haiku / v3 / 2026-09-20`); for every item and repetition, execute the step's real pipeline (not a copy); link the trace to the dataset item; compute the programmatic graders and write them as scores on the run. Run metadata: model, prompt@version, effort, git commit.
- [ ] 11.6 🔭 **Langfuse-managed judges.** Under *Evaluation → Evaluators* create an LLM-as-a-judge evaluator for `no_diagnosis` and one for `grounded_in_kb`, with your rubric, connected to Bedrock, filtered to experiment traces. Langfuse runs them by itself on every new matching trace. Compare with the alternative, a judge inside the runner: the former is convenient and reusable in production (Module 14), the latter is testable and versioned in git. You can keep both.
- [ ] 11.7 🔭 **Calibrating the judge.** Send 30–50 traces to an **annotation queue**, label them yourself (or with a clinical colleague) using the same scores the judge assigns, then compare judge-versus-human agreement. Below 90% on clear-cut cases, the rubric needs another pass. Repeat whenever you change the judge model.
- [ ] 11.8 💡 **Performance metrics from the API, not estimated.** Input, output and cache tokens from the response; cost computed from the price list of the model **that actually answered**; latency of the successful call only, excluding retries. The judge's cost is visible separately in Langfuse because its generations are distinct traces.
- [ ] 11.9 💡 **Repetitions and noise.** Run every case R times (at least 2–3). The half-width of the confidence interval on a success rate is roughly `1/sqrt(n·R)`: 30 cases × 2 repetitions ≈ ±13 points. If the difference between two runs is below the noise, **there is no difference**. The runner prints that number in the report; Langfuse shows per-run averages, not intervals, so compute them yourself.
- [ ] 11.10 💡 **Evaluation harness hygiene.** Separate infrastructure errors (timeouts, throttling, output truncated at `max_tokens`) from model errors: the former become an `infra_error = true` score and never enter the quality average. The full trajectory is already in Langfuse (the trace linked to the item). Verify that the model that answered is the one requested. Run an **oracle** (the expected answers must pass) and a **null baseline** (an empty answer must fail): if either misbehaves, the evaluation is broken.
- [ ] 11.11 🧪 🔭 **Model comparison for `SAFETY`.** Three runs (Haiku, Sonnet, a non-Anthropic model) on the same dataset and prompt. Use Langfuse's **run comparison** view for the dataset: average scores, cost and latency side by side. Then open the cases where the runs disagree: they are the most instructive. Decision in an ADR.
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
- [ ] 12.4 💡 **Level 3: contract tests against the real services (nightly, `#[ignore]`).** A handful of tests, one per capability declared in the `ModelRegistry`: the model answers; it supports tools; it supports structured output; cache reads work; streaming emits the expected events; a small `max_tokens` produces `stop_reason = max_tokens`; every parameter in `accepted_params` is accepted, and one it omits (such as `effort` on Haiku 4.5) is rejected when sent directly, bypassing the capability table, so you find out the day a provider changes the rules. Plus two for Langfuse: a trace sent over OTel appears through the API within N seconds, and a `production` prompt downloads. They run with `cargo nextest run --run-ignored ignored-only` and credentials supplied by CI through OIDC (never static access keys in secrets).
- [ ] 12.5 💡 🔭 **Level 4: regression experiments (nightly, or on prompt and model changes).** The nightly job runs the Module 11 runner over the regression datasets, then reads the run's scores through the API and compares them against **thresholds**: `safety.false_negative_rate ≤ 2%`, `generation.schema_valid ≥ 99%`, `generation.grounded ≥ 95%`, average cost per message ≤ X. The threshold must sit **above** the estimated noise, otherwise the test is a false-alarm generator. Failure means a red build, with a link to the Langfuse run in the message. A Langfuse **alert** on the run's average score (Module 14) is the second net.
- [ ] 12.6 💡 **Non-determinism.** In level 3 and 4 tests never assert on exact text: assert properties (valid schema, field present, value within a set), over several repetitions, above a threshold. Mark the tests prone to flakiness and track their failure rate: if it rises, that is a signal, not an annoyance.
- [ ] 12.7 💡 **Pinning and canaries.** Pin model versions where the provider allows it; when a new version arrives, run the full level 3 and 4 suite against it **before** changing the registry (that is your canary): in Langfuse the new run is compared against the last good one. Document the migration in an ADR.
- [ ] 12.8 Property-based tests with `proptest` for the hostile functions: the budgeted window with randomly sized turns, the structured output parser with almost-valid JSON, approximate tokenization.
- [ ] 12.9 Security as regression: an `injection-v1` dataset of prompt injection attempts that **must not** change `GENERATION` behaviour or bypass `SAFETY` (Module 13). It is a Langfuse dataset like any other.
- [ ] 12.10 Wire it all into CI: `ci.yml` runs levels 1–2; `nightly.yml` runs 3–4 and publishes the report with links to the runs; a badge in the README.

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
- [ ] 13.4 💡 🔭 **Personal data in prompts and in traces.** Minimize what enters the prompt; pseudonymize identifiers (the `user.id` in Langfuse must never be a real identifier); apply **PII redaction before export** in traces (the function from 5.10: names, phone numbers, emails, national identifiers; for free text consider a small model or the Bedrock Guardrails PII API); set **retention** per Langfuse project; document the data flow (who sees what, in which region). Bedrock does not use your data to train models and does not retain it beyond the request, except for logging you enable yourself: revisit the choice from 3.9.
- [ ] 13.5 💡 🔭 **Data residency.** Only `eu.*` inference profiles; no feature that routes outside the EU; Langfuse Cloud EU region or self-hosted in the EU (Module 15); verify where logs, traces, datasets and exports end up. Write it all in an ADR: it will be the basis of the legal review.
- [ ] 13.6 💡 **Least-privilege IAM.** Replace `AmazonBedrockFullAccess` with a policy scoped to the ARNs of the inference profiles you use; separate roles for the app, CI and the evaluation runner; no static access keys in production (IAM roles for ECS/Lambda, OIDC for GitHub Actions).
- [ ] 13.7 💡 🔭 **Secrets and Langfuse access.** Langfuse API keys in Secrets Manager, separate for the app, CI and the runner; in Langfuse, members with minimal roles (who can move `production` on prompts, who can see traces with content); the `production` label **protected**.
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
- [ ] 14.5 🔭 **Reconciled costs.** The cost Langfuse computes (from tokens and the prices in 5.4) should be compared monthly against the AWS bill filtered by the tag from 3.8 and by **Application Inference Profile** (create one application profile per environment: Cost Explorer then shows cost per profile). If they do not match within 5%, a price or a model id in the registry is wrong.
- [ ] 14.6 🔭 **Online evaluation.** The LLM-as-a-judge evaluators from Module 11 also run in production: configure them with a filter (`environment = prod`, sampling at say 5%, excluding irrelevant steps) so Langfuse assigns `no_diagnosis` and `grounded_in_kb` to a sample of real traffic. The judge's cost is measurable and must be kept in check through sampling. Also add **code-based** evaluators (for example: the reply cites only knowledge base ids that exist) where a model is not needed.
- [ ] 14.7 🔭 **User feedback.** The client sends thumbs up or down (and an optional comment) with the `trace_id` returned in 9.7; the app writes it as a `user_feedback` score on the trace through the API. In Langfuse, filter traces with negative feedback: they are the primary source of new dataset items (11.15).
- [ ] 14.8 🔭 **Continuous human review.** An annotation queue, "weekly clinical review", fed automatically with traces where `guardrail_pass = false`, where feedback is negative, or where the judge's score is below threshold. A person labels them; comparing human labels against the judge keeps the judge calibrated over time (11.7).
- [ ] 14.9 💡 **Drift.** Weekly, compare online scores and distributions (reply length, escalation rate, cost per message) against the previous release's baseline: a change with no deployment of your own is almost always a change in the model or in a prompt. Comparing dataset runs (12.7) confirms or refutes it.
- [ ] 14.10 💡 **The rest of the infrastructure.** Langfuse specializes in the LLM; CPU, memory, databases, queues and HTTP latency live elsewhere (in this company: Datadog). Two ways to correlate: (a) use the **OpenTelemetry Collector** as a fan-out, with the app sending traces to the collector and the collector forwarding to both Langfuse and Datadog (the same traces, with identical `trace_id`, in two places); (b) export aggregate metrics from Langfuse through the **Metrics API** into the existing monitoring system. Try (a): it is the pattern that lets you change backends without touching the app.
- [ ] 14.11 💡 🔭 **Langfuse volume and cost.** Estimate traces per month and observations per trace; check the Cloud plan's limits or size the self-hosted deployment; set retention; use **batch export** (to S3) for historical offline analysis.
- [ ] 14.12 Write a short runbook: "latency is rising" → what to look at in Langfuse and Datadog; "guardrails fire more often" → compare prompt version and model over the last hours; "cost has doubled" → cost-per-role dashboard, then *Users*; "the online score is dropping" → last regression run, last deployment, last label move.
- [ ] 14.13 🧪 Try debugging one conversation: from a `trace_id`, reconstruct every step, the exact prompts (with versions, clickable from the generation), the router's decisions and the cost. If you cannot do it in five minutes, something is missing from the instrumentation.

**Done when** a "Pipeline" dashboard exists with active alerts, an online evaluator runs on a sample of `prod`, user feedback arrives as scores, an annotation queue fills itself, and the runbook is written.

**Self-check.** Why is the `guardrail_pass` rate an excellent canary for silent provider changes? Why is sampling essential for online evaluation?

**Going deeper.** Langfuse docs: [LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge) (evaluators on live traffic), [Custom Dashboards](https://langfuse.com/docs/metrics/features/custom-dashboards), [Alerts](https://langfuse.com/docs/observability/features/alerts), [Annotation Queues](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues), [Metrics API](https://langfuse.com/docs/metrics/features/metrics-api), [Public API](https://langfuse.com/docs/api-and-data-platform/features/public-api).

---

## Module 15 · Deploying to production

⏱ 8–12 hours.

**Goal.** The application running on AWS with IAM roles rather than keys, per-environment configuration, working streaming, gradual rollout, cost controls; plus a reasoned decision about where Langfuse runs.

**Why it matters.** Many of the choices made so far (streaming, timeouts, caching, quotas, trace volume) can only be measured with a real deployment and some load.

### Steps

- [ ] 15.1 **Packaging.** A multi-stage Dockerfile for Rust (build in a toolchain image, run on `distroless` or `debian-slim`), an optimized binary (`--release`, LTO), a small image. Vulnerability scanning (`cargo deny`, an image scanner).
- [ ] 15.2 💡 **Twelve-factor configuration.** Everything from environment variables or Parameter Store: region, model ids per role, context budgets, router thresholds, Langfuse host and keys, tracing policy (content, sampling). No `if env == prod` in the code.
- [ ] 15.3 💡 **Where to run the app.** Evaluate three options and choose with an ADR: **ECS Fargate** (an always-on HTTP service, natural SSE streaming, the sensible default); **AWS Lambda** with `cargo-lambda` (great for spiky load; ⚠️ response streaming, timeouts and **flushing OTel traces** before the invocation ends all need verification); **Bedrock AgentCore Runtime** (a managed environment for agents, with a Gateway for tools, Memory, Policy with Guardrails and built-in Observability: useful for seeing what a managed runtime buys you).
- [ ] 15.4 💡 🔭 **Where to run Langfuse.** Two options, ADR required:
  - **Langfuse Cloud, EU region**: no operations, data in Ireland, compliance documented by the vendor; to be reviewed with legal as a sub-processor.
  - **Self-hosted on AWS EU**: `web` and `worker` on ECS, PostgreSQL on RDS, ClickHouse (self-managed on EC2/EKS, or ClickHouse Cloud in an EU region), Redis on ElastiCache, S3 for blobs and exports. Maximum control over the data, but upgrades, backups and capacity are yours.
  🧪 Do the **local** self-host with Docker Compose at least once, to touch the components and understand what running them involves.
- [ ] 15.5 **Identity.** An IAM role for the task or function with the least-privilege policy from Module 13. No credentials in the image. Langfuse keys from Secrets Manager.
- [ ] 15.6 **The server.** `axum` with health checks (`/healthz`, `/readyz`), graceful shutdown (finish in-flight requests, cancel background tasks sensibly, **flush the OTel exporter**), a per-request timeout, body size limits, per-user rate limiting, and CORS if needed.
- [ ] 15.7 **State.** SQLite is not enough for a multi-instance production: managed PostgreSQL (RDS) with `pgvector` for the knowledge base. Versioned migrations (`sqlx migrate`).
- [ ] 15.8 💡 🔭 **Gradual rollout.** For the app: two versions in parallel (blue/green, or a canary on 5% of traffic). For prompts: the active version is the `production` label in Langfuse, so a prompt rollback is a label move with no deployment; a `canary` label read by 5% of requests lets you try a new prompt on real traffic, with online scores (14.6) filtered by prompt version telling you whether to promote it. Keep the prompt cache short enough (5–10 minutes) for rollback to be fast.
- [ ] 15.9 💡 **Cost controls.** Budgets and alerts are already active (3.8, 14.4); add an emergency kill switch that degrades the service (escalation and fixed replies only) if hourly cost crosses a threshold.
- [ ] 15.10 🧪 **Load testing.** With `k6` or `oha`, 20–50 simulated users sending synthetic messages. Watch for: Bedrock throttling (the quotas from 3.7), p95 per step in Langfuse, router behaviour under stress, cost per minute, trace ingestion lag. Tune quotas, timeouts, the circuit breaker and the exporter's batch size.
- [ ] 15.11 Deployment pipeline: build on tag, push to ECR, deploy to ECS with manual approval for production; the Module 12 nightly must be green before promotion.
- [ ] 15.12 Operational runbook: how to roll back the app and a prompt, how to switch models in an emergency (the router override), how to turn off content tracing, who to call.

**Done when** the app answers in production with streaming, traces arrive in Langfuse with `environment = prod`, alerts are active, and you have rolled back a prompt at least once by moving a label.

**Self-check.** Why must the active prompt version be a label rather than code? What are the hidden costs of self-hosting Langfuse?

---

## Module 16 · Optional deep dives

To explore once the core is solid. Each one is a small experiment with a card in the Work Log.

- [ ] 16.1 **Agent frameworks compared.** Reimplement the Module 7 agent with **[Rig](https://github.com/0xPlaygrounds/rig)** ([`rig-core`](https://docs.rs/rig-core) + [`rig-bedrock`](https://docs.rs/rig-bedrock), the most mature Rust framework, with typed tools, RAG, MCP and OpenTelemetry attributes built in). Compare lines of code, control over the loop, testability, and observability in Langfuse. To see what a "complete" framework does, also look at the concepts in LangChain 1.x (Python: `create_agent`, middleware, LangGraph; start from [What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1) and [Agent middleware](https://www.langchain.com/blog/agent-middleware)) without switching stacks: your harness **is** an agent with hand-written middleware. ADR: "framework versus hand-written harness".
- [ ] 16.2 **Multi-agent (orchestrator-workers).** An orchestrator delegates sub-tasks to workers using a cheap model and isolated context, then synthesizes. In Langfuse the trace shows the whole tree. Compare against the single-model pipeline on a knowledge base research task with several questions.
- [ ] 16.3 **Tool search and on-demand loading.** With dozens of tools the list itself costs: techniques for exposing only the tools relevant to a turn (a local index, short descriptions with the full schema on request).
- [ ] 16.4 **Memory as a tool.** Give the model a `memory_read/write` tool over a directory or table and let it decide what to remember; compare with the structured profile from Module 8.
- [ ] 16.5 **Langfuse in more depth.** Composed prompts (a prompt that includes others, to reuse shared rules between `GENERATION` and `GUARDRAILS`); webhooks on prompt changes to trigger the nightly when someone moves `production`; **session-level** scores (the quality of a whole conversation rather than a single message); multimodal traces; batch export to S3 and offline analysis with DuckDB.
- [ ] 16.6 **Bedrock AgentCore in depth.** Runtime, Gateway (tools over MCP with authentication), Memory, Policy with Guardrails, Observability. How much of your code would it replace? Can its traces coexist with Langfuse?
- [ ] 16.7 **Claude Platform on AWS.** Run the same pipeline on Anthropic's managed access inside AWS (parity with the Anthropic API: server-side compaction, context editing, batch). An honest comparison with Bedrock on features, price and data residency.
- [ ] 16.8 **Batch inference.** For `PROFILE_UPDATE` and for evaluations, Bedrock's batch mode is cheaper but asynchronous: when is it worth it?
- [ ] 16.9 **Fine-tuning and distillation versus prompting.** When a small fine-tuned classifier beats a prompt on a large model (`SAFETY` is a candidate). Bedrock offers customization for some models; weigh cost and maintenance. Annotated Langfuse datasets are the starting point for the training set.
- [ ] 16.10 **Migrating to a new model.** Checklist: capabilities (contract tests), full experiments compared in Langfuse against the last good run, a prompt audit for cruft written for older models, recalibrating `effort`, costs, a production canary with online scores, a rollback ready.
- [ ] 16.11 **Cost optimization as a process.** The order of the levers: caching, input token hygiene (context, tool results), output hygiene, batch, then `effort`, then changing model. Always measure cost per **completed task**, which in Langfuse is cost per trace, not per generation.
- [ ] 16.12 **Multimodal and voice.** Image input (Converse supports image and document blocks), transcription and speech synthesis as pipeline steps.

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
- [Self-hosting](https://langfuse.com/self-hosting) (v3 architecture, Docker Compose, Helm)
- [Data regions](https://langfuse.com/security/data-regions) and [EU data residency and GDPR](https://langfuse.com/resources/engineering/langfuse-eu-data-residency-gdpr)
- [Changelog](https://langfuse.com/changelog) (features move fast)
- Rust crates from the genai-rs organization: [`opentelemetry-langfuse`](https://github.com/genai-rs/opentelemetry-langfuse), [`langfuse-ergonomic`](https://github.com/genai-rs/langfuse-ergonomic), [`langfuse-client-base`](https://github.com/genai-rs/langfuse-client-base)

**Rust**
- [Bedrock Runtime examples for Rust](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/rust_bedrock-runtime_code_examples.html)
- [Testing with the SDK](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/testing.html) (mocks and replay)
- [`aws-sdk-bedrockruntime` on docs.rs](https://docs.rs/aws-sdk-bedrockruntime)
- [Official MCP SDK (`rmcp`)](https://github.com/modelcontextprotocol/rust-sdk)
- [Rig](https://github.com/0xPlaygrounds/rig) and the [`rig-bedrock`](https://docs.rs/rig-bedrock) crate (optional comparison)
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
