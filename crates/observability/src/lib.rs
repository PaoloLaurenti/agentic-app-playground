//! Tracing for the whole application. Spans are printed on standard error and, when Langfuse is
//! configured, also exported to it as OpenTelemetry spans over OTLP/HTTP.
//!
//! The rest of the code only uses `tracing`: which backend receives the spans is decided here.

use std::collections::HashMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{Protocol, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// What the logs show when `RUST_LOG` is not set: `aws_config` at `info` prints the whole
/// credential chain on every run.
const DEFAULT_LOG_FILTER: &str = "info,aws_config=warn";

/// A Langfuse project, reached through its OpenTelemetry endpoint.
pub struct LangfuseConfig {
    /// The region's base URL, such as `https://cloud.langfuse.com`.
    pub host: String,
    pub public_key: String,
    pub secret_key: String,
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
    #[error("cannot build the OTLP exporter: {0}")]
    Exporter(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("cannot install the tracing subscriber: {0}")]
    Subscriber(#[from] tracing_subscriber::util::TryInitError),
    #[error("cannot send the last spans to Langfuse: {0}")]
    Flush(#[from] opentelemetry_sdk::error::OTelSdkError),
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
            .with_filter(targets)
    });

    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()?;
    Ok(Telemetry { provider })
}

fn tracer_provider(
    service_name: &'static str,
    langfuse: &LangfuseConfig,
) -> Result<SdkTracerProvider, Error> {
    let credentials = BASE64.encode(format!("{}:{}", langfuse.public_key, langfuse.secret_key));
    let headers = HashMap::from([
        ("Authorization".to_owned(), format!("Basic {credentials}")),
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
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name(service_name).build())
        .build())
}
