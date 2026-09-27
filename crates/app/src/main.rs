//! The command-line entry point of the playground.

mod pricing;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use llm_bedrock::BedrockClient;
use llm_core::{
    ContentBlock, LlmClient, LlmRequest, LlmResponse, Message, ModelId, Role, SystemBlock,
};

#[derive(Parser)]
#[command(about = "The agentic app playground")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Call a model and print the reply, stop reason, tokens, latency and estimated cost.
    Hello {
        /// Which model from `.env` to call.
        #[arg(long, value_enum, default_value_t = ModelRole::Fast)]
        model: ModelRole,
        /// The user message.
        #[arg(long, default_value = "Answer with a single word: ready?")]
        message: String,
        /// A file whose text becomes the system prompt, followed by a cache point.
        #[arg(long)]
        system: Option<PathBuf>,
        /// How many times to repeat the same call.
        #[arg(long, default_value_t = 1)]
        times: u32,
        /// The ceiling on output tokens.
        #[arg(long, default_value_t = 50)]
        max_tokens: u32,
    },
}

/// A model is chosen by its role, and `.env` says which model plays it.
#[derive(Clone, Copy, ValueEnum)]
enum ModelRole {
    Fast,
    Main,
}

impl ModelRole {
    fn env_var(self) -> &'static str {
        match self {
            ModelRole::Fast => "BEDROCK_MODEL_FAST",
            ModelRole::Main => "BEDROCK_MODEL_MAIN",
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    load_env_file()?;
    match Cli::parse().command {
        Command::Hello {
            model,
            message,
            system,
            times,
            max_tokens,
        } => hello(model, &message, system.as_deref(), times, max_tokens).await,
    }
}

/// Loads `.env` when it exists. Without it the environment must already hold the variables, as
/// it will in CI and in production.
fn load_env_file() -> Result<()> {
    match dotenvy::dotenv() {
        Ok(_) => Ok(()),
        Err(err) if err.not_found() => Ok(()),
        Err(err) => Err(err).context("cannot read .env"),
    }
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name)
        .with_context(|| format!("{name} is not set: copy .env.example to .env and fill it in"))
}

async fn hello(
    role: ModelRole,
    message: &str,
    system: Option<&Path>,
    times: u32,
    max_tokens: u32,
) -> Result<()> {
    let region = required_env("AWS_REGION")?;
    let model = required_env(role.env_var())?;
    // The EU-only rule of AGENTS.md, enforced where the configuration enters the program.
    ensure!(
        region.starts_with("eu-"),
        "AWS_REGION {region} is not an EU region"
    );
    ensure!(
        model.starts_with("eu."),
        "{} = {model} is not an EU inference profile",
        role.env_var()
    );
    ensure!(times > 0, "--times must be at least 1");

    let system = match system {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read {}", path.display()))?;
            vec![SystemBlock::Text(text), SystemBlock::CachePoint]
        }
        None => vec![],
    };
    let request = LlmRequest {
        model: ModelId::new(&model),
        system,
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text(message.to_owned())],
        }],
        max_tokens,
        extra: serde_json::Value::Null,
    };

    // Built once and reused by every call: loading the configuration and the credentials is the
    // expensive part.
    let client = BedrockClient::new(region).await;
    println!("model: {model}");

    let mut latencies = Vec::new();
    for call in 1..=times {
        let started = Instant::now();
        let response = client.complete(request.clone()).await?;
        let wall_time = started.elapsed();
        print_call(call, &model, &response, wall_time);
        latencies.push(response.latency);
    }
    if times > 1 {
        print_latency_summary(&latencies)?;
    }
    Ok(())
}

fn print_call(call: u32, model: &str, response: &LlmResponse, wall_time: Duration) {
    let reply: String = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            ContentBlock::CachePoint => None,
        })
        .collect();
    let usage = &response.usage;
    let cost = match pricing::price_of(model) {
        Some(price) => format!("{:.6} USD", pricing::cost(usage, &price)),
        None => "unknown, no price for this model".to_owned(),
    };

    println!("\ncall {call}");
    println!("  reply:       {reply}");
    println!("  stop reason: {:?}", response.stop_reason);
    println!(
        "  tokens:      {} input, {} output, {} cache read, {} cache write",
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens
    );
    println!(
        "  latency:     {} ms on Bedrock, {} ms end to end",
        response.latency.as_millis(),
        wall_time.as_millis()
    );
    println!("  cost:        {cost}");
}

/// Minimum, mean and maximum of the latency Bedrock reports, which excludes the network and the
/// client's own setup.
fn print_latency_summary(latencies: &[Duration]) -> Result<()> {
    let (Some(min), Some(max)) = (latencies.iter().min(), latencies.iter().max()) else {
        bail!("no latency to summarize");
    };
    let count = u32::try_from(latencies.len()).context("too many calls to summarize")?;
    let mean = latencies.iter().sum::<Duration>() / count;
    println!(
        "\nlatency over {count} calls: min {} ms, mean {} ms, max {} ms",
        min.as_millis(),
        mean.as_millis(),
        max.as_millis()
    );
    Ok(())
}
