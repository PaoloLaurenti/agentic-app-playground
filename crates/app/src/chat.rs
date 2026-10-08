//! `app chat`: a conversation read from standard input, one message per line. Every message is
//! one trace, and all the messages of a run share one session, so Langfuse shows the whole
//! conversation in order.

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Args;
use llm_core::{ContentBlock, LlmClient, LlmRequest, LlmResponse, Message, ModelId, Role};
use std::sync::Arc;

use observability::Scores;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{ModelRole, bedrock_target, output, pricing, system_prompt, text_of};

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
    println!("model {model}\nsession {session_id} · user {}\n", args.user);
    // A prompt only when a person is typing; piped messages need none.
    let prompt = std::io::stdin().is_terminal();

    // The model remembers nothing between calls: every turn sends the whole conversation so far.
    let mut messages = Vec::new();
    let mut total_cost = 0.0;
    // Read without blocking the runtime's thread while waiting for the next line.
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut turn_number = 0;
    loop {
        if prompt {
            print!("you › ");
            std::io::stdout().flush()?;
        }
        let Some(line) = lines.next_line().await? else {
            break;
        };
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if !prompt {
            println!("you › {text}");
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
            prompt: None,
            output_schema: None,
        };
        let response = turn(client.as_ref(), request, text, &session_id, &args.user).await?;

        let cost = price.map(|price| pricing::cost(&response.usage, &price));
        total_cost += cost.unwrap_or_default();
        println!(
            "{}",
            output::labelled("bot › ", &text_of(&response.message))
        );
        let usage = output::usage_line(&response.usage, cost);
        println!(
            "      {}\n",
            output::dimmed(&usage, output::color_enabled())
        );
        messages.push(response.message);
    }

    match price {
        Some(_) => println!("{turn_number} turns · ${total_cost:.6}"),
        None => println!("{turn_number} turns"),
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
