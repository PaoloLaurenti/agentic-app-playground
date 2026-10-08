//! The smoke check: did a model call produce a normal reply? Not whether the reply is good, only
//! whether there is one, like switching a device on to see whether it smokes.

use llm_core::{LlmRequest, LlmResponse, StopReason};
use observability::Verdict;

use crate::text_of;

/// `smoke_ok` passes when the model finished on its own and said something.
pub fn smoke_ok(_req: &LlmRequest, resp: &LlmResponse) -> Verdict {
    Verdict {
        name: "smoke_ok",
        value: resp.stop_reason == StopReason::EndTurn && !text_of(&resp.message).trim().is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use llm_core::{ContentBlock, Message, ModelId, Role, Usage};

    use super::*;

    fn request() -> LlmRequest {
        LlmRequest {
            model: ModelId::new("test-model"),
            system: vec![],
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text("Ready?".into())],
            }],
            max_tokens: 50,
            extra: serde_json::Value::Null,
            prompt: None,
            output_schema: None,
        }
    }

    fn response(stop_reason: StopReason, reply: &str) -> LlmResponse {
        LlmResponse {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(reply.into())],
            },
            stop_reason,
            usage: Usage::default(),
            latency: Duration::from_millis(500),
        }
    }

    #[test]
    fn a_reply_that_ends_its_turn_passes() {
        let verdict = smoke_ok(&request(), &response(StopReason::EndTurn, "Ready."));

        assert_eq!(
            verdict,
            Verdict {
                name: "smoke_ok",
                value: true
            }
        );
    }

    #[test]
    fn a_reply_cut_off_by_max_tokens_fails() {
        assert!(!smoke_ok(&request(), &response(StopReason::MaxTokens, "Rea")).value);
    }

    #[test]
    fn an_empty_reply_fails_even_when_it_ends_its_turn() {
        assert!(!smoke_ok(&request(), &response(StopReason::EndTurn, "  ")).value);
    }
}
