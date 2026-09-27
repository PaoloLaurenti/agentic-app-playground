//! The command-line entry point of the playground.

mod pricing;

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use futures::StreamExt;
use llm_bedrock::BedrockClient;
use llm_core::{
    ContentBlock, LlmClient, LlmEvent, LlmRequest, Message, ModelId, Role, StopReason, SystemBlock,
    Usage,
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
    Hello(HelloArgs),
}

#[derive(Args)]
struct HelloArgs {
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
    /// Print the reply as it arrives, and measure the time to first token.
    #[arg(long)]
    stream: bool,
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

/// What one call produced and how long it took.
struct Measure {
    stop_reason: StopReason,
    usage: Usage,
    /// The latency Bedrock reports, which excludes the network and the client's own setup.
    bedrock_latency: Duration,
    /// From sending the request to the first piece of text; only a stream has one.
    first_token: Option<Duration>,
    /// From sending the request to the last byte of the reply.
    end_to_end: Duration,
}

#[tokio::main]
async fn main() -> Result<()> {
    load_env_file()?;
    match Cli::parse().command {
        Command::Hello(args) => hello(args).await,
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

async fn hello(args: HelloArgs) -> Result<()> {
    let region = required_env("AWS_REGION")?;
    let model = required_env(args.model.env_var())?;
    // The EU-only rule of AGENTS.md, enforced where the configuration enters the program.
    ensure!(
        region.starts_with("eu-"),
        "AWS_REGION {region} is not an EU region"
    );
    ensure!(
        model.starts_with("eu."),
        "{} = {model} is not an EU inference profile",
        args.model.env_var()
    );
    ensure!(args.times > 0, "--times must be at least 1");

    let system = match &args.system {
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
            content: vec![ContentBlock::Text(args.message.clone())],
        }],
        max_tokens: args.max_tokens,
        extra: serde_json::Value::Null,
    };

    // Built once and reused by every call: loading the configuration and the credentials is the
    // expensive part.
    let client = BedrockClient::new(region).await;
    println!("model: {model}");

    let mut measures = Vec::new();
    for call in 1..=args.times {
        println!("\ncall {call}");
        let measure = if args.stream {
            call_streaming(&client, request.clone()).await?
        } else {
            call_complete(&client, request.clone()).await?
        };
        print_measure(&model, &measure);
        measures.push(measure);
    }
    if measures.len() > 1 {
        print_summary(&measures)?;
    }
    Ok(())
}

async fn call_complete(client: &dyn LlmClient, request: LlmRequest) -> Result<Measure> {
    let started = Instant::now();
    let response = client.complete(request).await?;
    let end_to_end = started.elapsed();

    let reply: String = response
        .message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            ContentBlock::CachePoint => None,
        })
        .collect();
    println!("  reply:       {reply}");

    Ok(Measure {
        stop_reason: response.stop_reason,
        usage: response.usage,
        bedrock_latency: response.latency,
        first_token: None,
        end_to_end,
    })
}

/// Prints the reply while it arrives. The stop reason comes before the token counts, so the
/// stream is read to its end.
async fn call_streaming(client: &dyn LlmClient, request: LlmRequest) -> Result<Measure> {
    let started = Instant::now();
    let mut events = client.stream(request).await?;

    let mut first_token = None;
    let mut stop_reason = None;
    let mut metadata = None;
    print!("  reply:       ");
    while let Some(event) = events.next().await {
        match event? {
            LlmEvent::TextDelta(text) => {
                first_token.get_or_insert_with(|| started.elapsed());
                print!("{text}");
                // Without a flush the terminal would show the text only at the end of the line.
                std::io::stdout().flush()?;
            }
            LlmEvent::Stop(reason) => stop_reason = Some(reason),
            LlmEvent::Metadata { usage, latency } => metadata = Some((usage, latency)),
        }
    }
    let end_to_end = started.elapsed();
    println!();

    let stop_reason = stop_reason.context("the stream ended without a stop reason")?;
    let (usage, bedrock_latency) = metadata.context("the stream ended without token counts")?;
    Ok(Measure {
        stop_reason,
        usage,
        bedrock_latency,
        first_token,
        end_to_end,
    })
}

fn print_measure(model: &str, measure: &Measure) {
    let usage = &measure.usage;
    let cost = match pricing::price_of(model) {
        Some(price) => format!("{:.6} USD", pricing::cost(usage, &price)),
        None => "unknown, no price for this model".to_owned(),
    };
    let first_token = match measure.first_token {
        Some(first_token) => format!(", first token after {} ms", first_token.as_millis()),
        None => String::new(),
    };

    println!("  stop reason: {:?}", measure.stop_reason);
    println!(
        "  tokens:      {} input, {} output, {} cache read, {} cache write",
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens
    );
    println!(
        "  latency:     {} ms on Bedrock, {} ms end to end{first_token}",
        measure.bedrock_latency.as_millis(),
        measure.end_to_end.as_millis(),
    );
    println!("  cost:        {cost}");
}

fn print_summary(measures: &[Measure]) -> Result<()> {
    println!("\nover {} calls:", measures.len());
    let bedrock: Vec<_> = measures.iter().map(|m| m.bedrock_latency).collect();
    let end_to_end: Vec<_> = measures.iter().map(|m| m.end_to_end).collect();
    let first_token: Vec<_> = measures.iter().filter_map(|m| m.first_token).collect();
    print_range("latency on Bedrock", &bedrock)?;
    print_range("end to end", &end_to_end)?;
    if !first_token.is_empty() {
        print_range("first token", &first_token)?;
    }
    Ok(())
}

/// Minimum, mean and maximum of a set of durations.
fn print_range(label: &str, durations: &[Duration]) -> Result<()> {
    let (Some(min), Some(max)) = (durations.iter().min(), durations.iter().max()) else {
        bail!("no {label} to summarize");
    };
    let count = u32::try_from(durations.len()).context("too many calls to summarize")?;
    let mean = durations.iter().sum::<Duration>() / count;
    println!(
        "  {label:<19} min {} ms, mean {} ms, max {} ms",
        min.as_millis(),
        mean.as_millis(),
        max.as_millis()
    );
    Ok(())
}
