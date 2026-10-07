//! A streamed reply arrives in pieces; the decorators that need the whole of it gather the pieces
//! here and rebuild the response when the stream ends.

use std::time::Duration;

use llm_core::{ContentBlock, LlmEvent, LlmResponse, Message, Role, StopReason, Usage};

/// What a stream has said so far.
#[derive(Default)]
pub(crate) struct ReplySoFar {
    text: String,
    started: bool,
    stop_reason: Option<StopReason>,
    metadata: Option<(Usage, Duration)>,
}

impl ReplySoFar {
    /// Adds an event; true when it is the first piece of text, the time to first token.
    pub(crate) fn add(&mut self, event: &LlmEvent) -> bool {
        match event {
            LlmEvent::TextDelta(text) => {
                self.text.push_str(text);
                !std::mem::replace(&mut self.started, true)
            }
            LlmEvent::Stop(reason) => {
                self.stop_reason = Some(reason.clone());
                false
            }
            LlmEvent::Metadata { usage, latency } => {
                self.metadata = Some((*usage, *latency));
                false
            }
        }
    }

    /// The whole response, once the stop reason and the token counts have both arrived.
    pub(crate) fn into_response(self) -> Option<LlmResponse> {
        let (usage, latency) = self.metadata?;
        Some(LlmResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(self.text)],
            },
            stop_reason: self.stop_reason?,
            usage,
            latency,
        })
    }
}
