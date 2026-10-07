//! What the commands print: one short line of measures under each reply, and errors in one line.

use llm_core::Usage;

use crate::Measure;

/// Tokens and cost of one call: `15 in / 5 out tokens · $0.000044`.
pub fn usage_line(usage: &Usage, cost: Option<f64>) -> String {
    let mut parts = vec![format!(
        "{} in / {} out tokens",
        usage.input_tokens, usage.output_tokens
    )];
    if usage.cache_read_tokens > 0 || usage.cache_write_tokens > 0 {
        parts.push(format!(
            "cache {} read, {} written",
            usage.cache_read_tokens, usage.cache_write_tokens
        ));
    }
    parts.push(cost.map_or_else(|| "no price".to_owned(), |cost| format!("${cost:.6}")));
    parts.join(" · ")
}

/// Everything `hello` measured on one call, after the reply.
pub fn measure_line(measure: &Measure, cost: Option<f64>) -> String {
    let line = format!(
        "{:?} · {} · Bedrock {} ms · total {} ms",
        measure.stop_reason,
        usage_line(&measure.usage, cost),
        measure.bedrock_latency.as_millis(),
        measure.end_to_end.as_millis(),
    );
    match measure.first_token {
        Some(first_token) => format!("{line} · first token {} ms", first_token.as_millis()),
        None => line,
    }
}

/// An error as one readable line, with what to do about it when that is known.
pub fn error_message(err: &anyhow::Error, aws_profile: &str) -> String {
    let text = err.to_string();
    if text.contains("Session token not found or invalid") {
        return format!("AWS credentials expired, run: aws sso login --profile {aws_profile}");
    }
    // The AWS SDK appends its whole error structure in parentheses after the message, and some
    // errors go on with one line per credential provider it tried.
    let text = text.lines().next().unwrap_or_default();
    match text.find(" (") {
        Some(start) if text.ends_with(')') => text[..start].to_owned(),
        _ => text.to_owned(),
    }
}

/// A message under a label such as `bot › `, its following lines aligned under the first.
pub fn labelled(label: &str, text: &str) -> String {
    let indent = " ".repeat(label.chars().count());
    let mut lines = text.lines();
    let first = format!("{label}{}", lines.next().unwrap_or_default());
    std::iter::once(first)
        // Blank lines stay empty, without trailing spaces.
        .chain(lines.map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{indent}{line}")
            }
        }))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Secondary text, such as the measures under a reply, in grey when `color` is on.
pub fn dimmed(text: &str, color: bool) -> String {
    // The ANSI codes for "faint" and "reset", which every common terminal understands.
    if color {
        format!("\x1b[2m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

/// Whether to use colors: only on a terminal, and never when `NO_COLOR` is set (no-color.org).
pub fn color_enabled() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use llm_core::StopReason;

    use super::*;

    fn measure(first_token: Option<Duration>) -> Measure {
        Measure {
            reply: "Ready.".into(),
            stop_reason: StopReason::EndTurn,
            usage: usage(15, 5, 0, 0),
            bedrock_latency: Duration::from_millis(599),
            first_token,
            end_to_end: Duration::from_millis(1456),
        }
    }

    #[test]
    fn a_call_is_summed_up_in_one_line() {
        assert_eq!(
            measure_line(&measure(None), Some(0.000044)),
            "EndTurn · 15 in / 5 out tokens · $0.000044 · Bedrock 599 ms · total 1456 ms"
        );
    }

    #[test]
    fn a_stream_also_shows_its_time_to_first_token() {
        assert!(
            measure_line(&measure(Some(Duration::from_millis(612))), Some(0.000044))
                .ends_with(" · total 1456 ms · first token 612 ms")
        );
    }

    fn usage(input: u32, output: u32, cache_read: u32, cache_write: u32) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: cache_read,
            cache_write_tokens: cache_write,
        }
    }

    #[test]
    fn tokens_and_cost_fit_in_one_line() {
        assert_eq!(
            usage_line(&usage(15, 5, 0, 0), Some(0.000044)),
            "15 in / 5 out tokens · $0.000044"
        );
    }

    #[test]
    fn the_cache_shows_only_when_it_was_used() {
        assert_eq!(
            usage_line(&usage(15, 5, 2184, 0), Some(0.000937)),
            "15 in / 5 out tokens · cache 2184 read, 0 written · $0.000937"
        );
    }

    #[test]
    fn a_model_without_a_price_says_so() {
        assert_eq!(
            usage_line(&usage(15, 5, 0, 0), None),
            "15 in / 5 out tokens · no price"
        );
    }

    #[test]
    fn expired_aws_credentials_say_how_to_log_in_again() {
        let err = anyhow::anyhow!(
            "dispatch failure: other: an error occurred while loading credentials: service error: \
             UnauthorizedException: Session token not found or invalid (DispatchFailure(…))"
        );

        assert_eq!(
            error_message(&err, "bedrock-playground"),
            "AWS credentials expired, run: aws sso login --profile bedrock-playground"
        );
    }

    #[test]
    fn the_sdks_debug_dump_after_the_message_is_dropped() {
        let err = anyhow::anyhow!(
            "invalid request: ValidationException: The provided model identifier is invalid. \
             (ValidationError(ValidationException {{ message: Some(\"…\") }}))"
        );

        assert_eq!(
            error_message(&err, "bedrock-playground"),
            "invalid request: ValidationException: The provided model identifier is invalid."
        );
    }

    #[test]
    fn a_reply_on_several_lines_stays_aligned_under_its_label() {
        assert_eq!(
            labelled("bot › ", "Ti chiami Mario!\n\nMe l'hai detto tu."),
            "bot › Ti chiami Mario!\n\n      Me l'hai detto tu."
        );
    }

    #[test]
    fn an_error_spread_over_several_lines_keeps_only_its_first() {
        let err = anyhow::anyhow!(
            "dispatch failure: no credentials could be loaded:\n  Profile: no such profile\n  Environment: not set"
        );

        assert_eq!(
            error_message(&err, "bedrock-playground"),
            "dispatch failure: no credentials could be loaded:"
        );
    }

    #[test]
    fn on_a_terminal_secondary_text_is_grey() {
        assert_eq!(
            dimmed("10 in / 43 out tokens", true),
            "\x1b[2m10 in / 43 out tokens\x1b[0m"
        );
    }

    #[test]
    fn without_colors_secondary_text_stays_plain() {
        assert_eq!(
            dimmed("10 in / 43 out tokens", false),
            "10 in / 43 out tokens"
        );
    }
}
