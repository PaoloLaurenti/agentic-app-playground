//! Tracing for the whole application. Spans are printed on standard error and, when Langfuse is
//! configured, also exported to it as OpenTelemetry spans over OTLP/HTTP.
//!
//! The rest of the code only uses `tracing`: which backend receives the spans is decided here.

mod builder;
mod generation;
mod reply_so_far;
mod scores;
mod scoring;
mod trace_context;
mod traced_client;

use std::collections::HashMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use trace_context::{TraceContextProcessor, is_valid_environment};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

pub use builder::LlmClientBuilder;
pub use scores::{ScoreClient, Scores, current_trace_id};
pub use scoring::{Evaluator, Verdict};

/// What the logs show when `RUST_LOG` is not set: only warnings and errors, so that the commands'
/// own output stays readable. `RUST_LOG=info` adds one line per closed span.
const DEFAULT_LOG_FILTER: &str = "warn";

/// A Langfuse project, reached through its OpenTelemetry endpoint.
pub struct LangfuseConfig {
    /// The region's base URL, such as `https://cloud.langfuse.com`.
    pub host: String,
    pub public_key: String,
    pub secret_key: String,
    /// Keeps test traces apart from production ones, such as `local` or `production`.
    pub environment: String,
    /// The application's version, to compare traces before and after a change.
    pub release: String,
}

pub struct Config<'a> {
    /// Shown in Langfuse as the `service.name` resource attribute.
    pub service_name: &'static str,
    /// The targets, usually crate names, whose spans are exported. Everything else, such as the
    /// AWS SDK's own spans, is only printed.
    pub exported_targets: &'a [&'a str],
    /// `None` prints the spans without exporting them, as in CI.
    pub langfuse: Option<LangfuseConfig>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "environment {0:?} is not valid for Langfuse: lowercase letters, digits, - and _, at most 40, not starting with langfuse"
    )]
    InvalidEnvironment(String),
    #[error("cannot build the OTLP exporter: {0}")]
    Exporter(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("cannot install the tracing subscriber: {0}")]
    Subscriber(#[from] tracing_subscriber::util::TryInitError),
    #[error("cannot send the last spans to Langfuse: {0}")]
    Flush(#[from] opentelemetry_sdk::error::OTelSdkError),
    #[error("cannot reach the Langfuse API: {0}")]
    Api(#[from] reqwest::Error),
    #[error("Langfuse rejected the score with status {status}: {reason}")]
    ScoreRejected {
        status: reqwest::StatusCode,
        reason: String,
    },
}

/// Holds the exporter. Spans leave in batches from a background thread, so a short-lived
/// program must call [`Telemetry::shutdown`] before exiting, or it loses the last batch.
#[must_use = "without shutdown the last spans are never sent"]
pub struct Telemetry {
    provider: Option<SdkTracerProvider>,
}

impl Telemetry {
    /// Sends the spans still buffered and stops the exporter.
    pub fn shutdown(self) -> Result<(), Error> {
        match self.provider {
            Some(provider) => Ok(provider.shutdown()?),
            None => Ok(()),
        }
    }
}

/// Installs the global subscriber: one layer prints, filtered by `RUST_LOG`, and one exports to
/// Langfuse, filtered by [`Config::exported_targets`].
pub fn init(config: Config) -> Result<Telemetry, Error> {
    let log_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));
    // Every span prints one line when it closes, with its fields. Standard error keeps the reply
    // on standard output clean.
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_span_events(FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .with_filter(log_filter);

    let provider = config
        .langfuse
        .map(|langfuse| tracer_provider(config.service_name, &langfuse))
        .transpose()?;
    let otel_layer = provider.as_ref().map(|provider| {
        let targets = Targets::new().with_targets(
            config
                .exported_targets
                .iter()
                .map(|target| (*target, LevelFilter::INFO)),
        );
        tracing_opentelemetry::layer()
            .with_tracer(provider.tracer(config.service_name))
            // Source location, thread, busy and idle time and target would only clutter the
            // metadata of every observation in Langfuse.
            .with_location(false)
            .with_threads(false)
            .with_tracked_inactivity(false)
            .with_target(false)
            .with_filter(targets)
    });

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()?;
    Ok(Telemetry { provider })
}

/// The `Authorization` header for a Langfuse project: Basic authentication with the public key as
/// the user and the secret key as the password. The OTLP endpoint and the public API share it.
fn basic_authorization(langfuse: &LangfuseConfig) -> String {
    let credentials = BASE64.encode(format!("{}:{}", langfuse.public_key, langfuse.secret_key));
    format!("Basic {credentials}")
}

fn tracer_provider(
    service_name: &'static str,
    langfuse: &LangfuseConfig,
) -> Result<SdkTracerProvider, Error> {
    if !is_valid_environment(&langfuse.environment) {
        return Err(Error::InvalidEnvironment(langfuse.environment.clone()));
    }
    let headers = HashMap::from([
        ("Authorization".to_owned(), basic_authorization(langfuse)),
        // Without it, spans sent straight over OTLP can take up to ten minutes to show up.
        ("x-langfuse-ingestion-version".to_owned(), "4".to_owned()),
    ]);
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        // Given in full: a programmatic endpoint is used as it is, without `/v1/traces` appended.
        .with_endpoint(format!(
            "{}/api/public/otel/v1/traces",
            langfuse.host.trim_end_matches('/')
        ))
        .with_headers(headers)
        .build()?;
    Ok(SdkTracerProvider::builder()
        .with_span_processor(TraceContextProcessor::new(
            langfuse.environment.clone(),
            langfuse.release.clone(),
        ))
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name(service_name).build())
        .build())
}

#[cfg(test)]
mod test_support {
    use std::collections::HashMap;
    use std::time::Duration;

    use futures::StreamExt;
    use futures::stream::BoxStream;
    use llm_core::{
        ContentBlock, LlmClient, LlmError, LlmEvent, LlmRequest, LlmResponse, Message, ModelId,
        Role, StopReason, Usage,
    };

    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::{
        InMemorySpanExporter, SdkTracerProvider, TracerProviderBuilder,
    };
    use tracing_subscriber::layer::SubscriberExt;

    /// A span as it would have left for Langfuse.
    pub struct ExportedSpan {
        pub name: String,
        pub trace_id: String,
        pub duration: Duration,
        pub attributes: HashMap<String, String>,
        /// The names of the events logged inside the span, such as warnings.
        pub events: Vec<String>,
    }

    /// Runs `f` with an OpenTelemetry layer that exports to memory instead of the network, and
    /// returns the spans it closed.
    pub fn exported_spans(f: impl FnOnce()) -> Vec<ExportedSpan> {
        exported_spans_through(SdkTracerProvider::builder(), f)
    }

    /// The same, with a provider that may already hold span processors of its own.
    pub fn exported_spans_through(
        provider: TracerProviderBuilder,
        f: impl FnOnce(),
    ) -> Vec<ExportedSpan> {
        let exporter = InMemorySpanExporter::default();
        let provider = provider.with_simple_exporter(exporter.clone()).build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        tracing::subscriber::with_default(subscriber, f);
        exporter
            .get_finished_spans()
            .unwrap()
            .into_iter()
            .map(|span| ExportedSpan {
                name: span.name.to_string(),
                trace_id: span.span_context.trace_id().to_string(),
                duration: span.end_time.duration_since(span.start_time).unwrap(),
                events: span
                    .events
                    .iter()
                    .map(|event| event.name.to_string())
                    .collect(),
                attributes: span
                    .attributes
                    .iter()
                    .map(|kv| (kv.key.to_string(), kv.value.to_string()))
                    .collect(),
            })
            .collect()
    }

    /// Runs a future to completion. Not with `futures::executor::block_on`: the in-memory
    /// exporter uses that one itself when a span closes, and the two cannot be nested.
    pub fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(future)
    }

    pub fn request() -> LlmRequest {
        LlmRequest {
            model: ModelId::new("test-model"),
            system: vec![],
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text("Ready?".into())],
            }],
            max_tokens: 50,
            extra: serde_json::Value::Null,
        }
    }

    pub fn reply() -> LlmResponse {
        LlmResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text("Ready.".into())],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 15,
                output_tokens: 5,
                ..Usage::default()
            },
            latency: Duration::from_millis(700),
        }
    }

    /// Answers with a fixed reply, or fails, without any provider behind it.
    pub struct FakeClient {
        pub fail: bool,
    }

    #[async_trait::async_trait]
    impl LlmClient for FakeClient {
        async fn complete(&self, _req: LlmRequest) -> Result<LlmResponse, LlmError> {
            if self.fail {
                return Err(LlmError::Throttled("slow down".into()));
            }
            Ok(reply())
        }

        async fn stream(
            &self,
            _req: LlmRequest,
        ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
            let reply = reply();
            let events = vec![
                Ok(LlmEvent::TextDelta("Rea".into())),
                Ok(LlmEvent::TextDelta("dy.".into())),
                Ok(LlmEvent::Stop(reply.stop_reason)),
                Ok(LlmEvent::Metadata {
                    usage: reply.usage,
                    latency: reply.latency,
                }),
            ];
            Ok(futures::stream::iter(events).boxed())
        }
    }

    /// Keeps the scores it is asked to send instead of sending them; can also fail or be slow.
    #[derive(Default)]
    pub struct RecordedScores {
        pub sent: std::sync::Mutex<Vec<(String, String, bool)>>,
        pub fail: bool,
        pub delay: Duration,
    }

    #[async_trait::async_trait]
    impl crate::Scores for RecordedScores {
        async fn boolean(
            &self,
            trace_id: &str,
            name: &str,
            value: bool,
        ) -> Result<String, crate::Error> {
            tokio::time::sleep(self.delay).await;
            if self.fail {
                return Err(crate::Error::InvalidEnvironment("a fake failure".into()));
            }
            self.sent
                .lock()
                .unwrap()
                .push((trace_id.to_owned(), name.to_owned(), value));
            Ok("score-1".to_owned())
        }
    }
}
