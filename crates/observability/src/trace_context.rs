//! The attributes that describe a whole trace, copied onto every span of it. Langfuse v4 filters
//! and aggregates observations one by one, so an attribute only on the root is invisible when
//! filtering its children.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use opentelemetry::trace::{Span as _, SpanId, TraceContextExt, TraceId};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{Span, SpanData, SpanProcessor};

/// Adds the environment, the release and the trace name to every span when it starts. The
/// trace name is the root span's own name, so the code that opens spans does nothing for it.
#[derive(Debug)]
pub(crate) struct TraceContextProcessor {
    environment: String,
    release: String,
    /// The name of every trace whose root span is still open.
    trace_names: Mutex<HashMap<TraceId, String>>,
}

impl TraceContextProcessor {
    pub(crate) fn new(environment: String, release: String) -> Self {
        Self {
            environment,
            release,
            trace_names: Mutex::new(HashMap::new()),
        }
    }
}

impl SpanProcessor for TraceContextProcessor {
    fn on_start(&self, span: &mut Span, cx: &Context) {
        span.set_attribute(KeyValue::new(
            "langfuse.environment",
            self.environment.clone(),
        ));
        span.set_attribute(KeyValue::new("langfuse.release", self.release.clone()));

        let trace_id = span.span_context().trace_id();
        let is_root = !cx.span().span_context().is_valid();
        // A poisoned lock only means another thread panicked while holding it; the map is
        // still usable, and tracing must never take the program down.
        let mut trace_names = self
            .trace_names
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let name = if is_root {
            let name = span.exported_data().map(|data| data.name.into_owned());
            if let Some(name) = &name {
                trace_names.insert(trace_id, name.clone());
            }
            name
        } else {
            trace_names.get(&trace_id).cloned()
        };
        if let Some(name) = name {
            span.set_attribute(KeyValue::new("langfuse.trace.name", name));
        }
    }

    fn on_end(&self, span: SpanData) {
        if span.parent_span_id == SpanId::INVALID {
            self.trace_names
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&span.span_context.trace_id());
        }
    }

    fn force_flush(&self) -> OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

/// Langfuse accepts lowercase letters, digits, `-` and `_`, at most 40 characters, not starting
/// with `langfuse`; anything else would not be filed under the intended environment.
pub(crate) fn is_valid_environment(name: &str) -> bool {
    (1..=40).contains(&name.len())
        && !name.starts_with("langfuse")
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use opentelemetry_sdk::trace::SdkTracerProvider;

    use super::*;
    use crate::test_support::exported_spans_through;

    #[test]
    fn every_span_of_a_trace_carries_its_environment_release_and_name() {
        let provider = SdkTracerProvider::builder()
            .with_span_processor(TraceContextProcessor::new("local".into(), "0.1.0".into()));

        let spans = exported_spans_through(provider, || {
            tracing::info_span!("hello").in_scope(|| {
                let _child = tracing::info_span!("call-model").entered();
            });
        });

        assert_eq!(spans.len(), 2);
        for span in &spans {
            assert_eq!(
                span.attributes["langfuse.environment"], "local",
                "{}",
                span.name
            );
            assert_eq!(
                span.attributes["langfuse.release"], "0.1.0",
                "{}",
                span.name
            );
            assert_eq!(
                span.attributes["langfuse.trace.name"], "hello",
                "{}",
                span.name
            );
        }
    }

    #[test]
    fn two_traces_keep_their_own_names() {
        let provider = SdkTracerProvider::builder()
            .with_span_processor(TraceContextProcessor::new("local".into(), "0.1.0".into()));

        let spans = exported_spans_through(provider, || {
            tracing::info_span!("hello").in_scope(|| {
                let _child = tracing::info_span!("call-model").entered();
            });
            tracing::info_span!("chat").in_scope(|| {
                let _child = tracing::info_span!("call-model").entered();
            });
        });

        let names: Vec<_> = spans
            .iter()
            .map(|span| span.attributes["langfuse.trace.name"].as_str())
            .collect();
        assert_eq!(names, ["hello", "hello", "chat", "chat"]);
    }

    #[test]
    fn environment_names_follow_the_langfuse_rules() {
        assert!(is_valid_environment("local"));
        assert!(is_valid_environment("staging-eu_1"));
        assert!(!is_valid_environment(""));
        assert!(!is_valid_environment("Local"));
        assert!(!is_valid_environment("langfuse-test"));
        assert!(!is_valid_environment(&"a".repeat(41)));
    }
}
