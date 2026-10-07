//! `app chat`: a conversation read from standard input, one message per line. Every message is
//! one trace, and all the messages of a run share one session, so Langfuse shows the whole
//! conversation in order.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Args;
use llm_core::{ContentBlock, LlmClient, LlmRequest, LlmResponse, Message, ModelId, Role};
use std::sync::Arc;

use observability::Scores;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{ModelRole, bedrock_target, pricing, system_prompt, text_of};

#[derive(Args)]
pub struct ChatArgs {
    /// Which model from `.env` to talk to.
    #[arg(long, value_enum, default_value_t = ModelRole::Fast)]
    model: ModelRole,
    /// A file whose text becomes the system prompt, followed by a cache point.
    #[arg(long)]
    system: Option<PathBuf>,
    /// The ceiling on output tokens of every reply.
    #[arg(long, default_value_t = 300)]
    max_tokens: u32,
    /// Who is talking: a synthetic id, never a real person's.
    #[arg(long, default_value = "synthetic-user-1")]
    user: String,
}

pub async fn chat(args: ChatArgs, scores: Option<Arc<dyn Scores>>) -> Result<()> {
    let (region, model) = bedrock_target(args.model)?;
    let system = system_prompt(args.system.as_deref())?;
    let client = crate::llm_client(region, scores).await;
    // One run of the command is one session.
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("the clock is before 1970")?;
    let session_id = format!("chat-{}", started.as_millis());
    let price = pricing::price_of(&model);
    println!("model: {model}\nsession: {session_id}\nuser: {}", args.user);

    // The model remembers nothing between calls: every turn sends the whole conversation so far.
    let mut messages = Vec::new();
    let mut total_cost = 0.0;
    // Read without blocking the runtime's thread while waiting for the next line.
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut turn_number = 0;
    while let Some(line) = lines.next_line().await? {
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        turn_number += 1;
        messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text(text.to_owned())],
        });
        let request = LlmRequest {
            model: ModelId::new(&model),
            system: system.clone(),
            messages: messages.clone(),
            max_tokens: args.max_tokens,
            extra: serde_json::Value::Null,
        };
        let response = turn(client.as_ref(), request, text, &session_id, &args.user).await?;

        let usage = &response.usage;
        println!("\nturn {turn_number}");
        println!("  you:    {text}");
        println!("  reply:  {}", text_of(&response.message));
        println!(
            "  tokens: {} input, {} output, {} cache read, {} cache write",
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens
        );
        if let Some(price) = &price {
            let cost = pricing::cost(usage, price);
            total_cost += cost;
            println!("  cost:   {cost:.6} USD");
        }
        messages.push(response.message);
    }

    println!("\n{turn_number} turns");
    if price.is_some() {
        println!("total cost: {total_cost:.6} USD");
    }
    Ok(())
}

/// One turn is one trace. Its root span carries the message and the reply, like `hello`, and also
/// the session and the user as the standard `session.id` and `user.id` fields, which every span of
/// the trace inherits.
#[tracing::instrument(
    name = "chat",
    skip_all,
    fields(
        session.id = session_id,
        user.id = user_id,
        input.value = message,
        output.value = tracing::field::Empty,
    )
)]
async fn turn(
    client: &dyn LlmClient,
    request: LlmRequest,
    message: &str,
    session_id: &str,
    user_id: &str,
) -> Result<LlmResponse> {
    let response = client.complete(request).await?;
    tracing::Span::current().record("output.value", text_of(&response.message).as_str());
    Ok(response)
}
