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

/// Adds the environment and the release to every span when it starts, and copies the root
/// span's name, session and user onto every span below it. The root declares the session and the
/// user as the standard `session.id` and `user.id` fields, so the code that opens spans never
/// names a Langfuse attribute.
#[derive(Debug)]
pub(crate) struct TraceContextProcessor {
    environment: String,
    release: String,
    /// What every trace whose root span is still open shares with its spans.
    traces: Mutex<HashMap<TraceId, TraceWide>>,
}

/// The attributes a root span hands down to the rest of its trace.
#[derive(Debug, Clone)]
struct TraceWide {
    name: String,
    session_id: Option<String>,
    user_id: Option<String>,
}

impl TraceWide {
    fn of_root(root: &SpanData) -> Self {
        let attribute = |key: &str| {
            root.attributes
                .iter()
                .find(|kv| kv.key.as_str() == key)
                .map(|kv| kv.value.to_string())
        };
        Self {
            name: root.name.to_string(),
            session_id: attribute("session.id"),
            user_id: attribute("user.id"),
        }
    }
}

impl TraceContextProcessor {
    pub(crate) fn new(environment: String, release: String) -> Self {
        Self {
            environment,
            release,
            traces: Mutex::new(HashMap::new()),
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
        let mut traces = self
            .traces
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let trace_wide = if is_root {
            let trace_wide = span.exported_data().map(|data| TraceWide::of_root(&data));
            if let Some(trace_wide) = &trace_wide {
                traces.insert(trace_id, trace_wide.clone());
            }
            trace_wide
        } else {
            traces.get(&trace_id).cloned()
        };
        if let Some(trace_wide) = trace_wide {
            span.set_attribute(KeyValue::new("langfuse.trace.name", trace_wide.name));
            if let Some(session_id) = trace_wide.session_id {
                span.set_attribute(KeyValue::new("langfuse.session.id", session_id));
            }
            if let Some(user_id) = trace_wide.user_id {
                span.set_attribute(KeyValue::new("langfuse.user.id", user_id));
            }
        }
    }

    fn on_end(&self, span: SpanData) {
        if span.parent_span_id == SpanId::INVALID {
            self.traces
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
    fn the_root_hands_its_session_and_user_down_to_every_span() {
        let provider = SdkTracerProvider::builder()
            .with_span_processor(TraceContextProcessor::new("local".into(), "0.1.0".into()));

        let spans = exported_spans_through(provider, || {
            tracing::info_span!("chat", session.id = "chat-1", user.id = "synthetic-user-1")
                .in_scope(|| {
                    let _child = tracing::info_span!("call-model").entered();
                });
        });

        assert_eq!(spans.len(), 2);
        for span in &spans {
            assert_eq!(
                span.attributes["langfuse.session.id"], "chat-1",
                "{}",
                span.name
            );
            assert_eq!(
                span.attributes["langfuse.user.id"], "synthetic-user-1",
                "{}",
                span.name
            );
        }
    }

    #[test]
    fn a_trace_without_session_or_user_gets_neither() {
        let provider = SdkTracerProvider::builder()
            .with_span_processor(TraceContextProcessor::new("local".into(), "0.1.0".into()));

        let spans = exported_spans_through(provider, || {
            let _root = tracing::info_span!("hello").entered();
        });

        assert!(!spans[0].attributes.contains_key("langfuse.session.id"));
        assert!(!spans[0].attributes.contains_key("langfuse.user.id"));
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
