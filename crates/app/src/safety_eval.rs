//! `app safety-eval`: runs the safety classifier compiled into the binary over a dataset of
//! messages labelled by hand, and reports how it did (6.10). Module 11 turns it into a Langfuse
//! dataset run.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Args;
use llm_core::{LlmClient, LlmRequest, ModelId, StopReason, Usage};
use observability::Scores;
use prompts::{Prompt, SafetyLabel, SafetyVerdict};
use serde::Deserialize;
use serde_json::Value;

use crate::{ModelRole, bedrock_target, pricing, text_of};

const SAFETY_V1: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../datasets/safety/v1.jsonl"
);

/// The labels in the order the report prints them.
const LABELS: [SafetyLabel; 3] = [
    SafetyLabel::Safe,
    SafetyLabel::PsychCrisis,
    SafetyLabel::MedicalEmergency,
];

#[derive(Args)]
pub struct SafetyEvalArgs {
    /// The dataset, one case per line.
    #[arg(long, default_value = SAFETY_V1)]
    dataset: PathBuf,
}

/// Runs every case of the dataset in order, one trace per case, all in one session, then prints
/// the report.
pub async fn safety_eval(args: SafetyEvalArgs, scores: Option<Arc<dyn Scores>>) -> Result<()> {
    let prompt = prompts::safety_classifier()?;
    let settings = CallSettings::from_config(&prompt.config)?;
    let (region, model) = bedrock_target(settings.role)?;
    let text = std::fs::read_to_string(&args.dataset)
        .with_context(|| format!("cannot read {}", args.dataset.display()))?;
    let cases = read_cases(&text)?;
    let client = crate::llm_client(region, scores).await;
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("the clock is before 1970")?;
    let session_id = format!("safety-eval-{}", started.as_millis());
    println!(
        "model {model} · prompt {} v{} · {} cases\nsession {session_id}\n",
        prompt.name,
        prompt.version,
        cases.len()
    );
    let classifier = Classifier {
        prompt,
        model: ModelId::new(&model),
        max_tokens: settings.max_tokens,
    };

    let mut outcomes = Vec::new();
    for case in &cases {
        let outcome = traced_classify(&classifier, client.as_ref(), case, &session_id).await?;
        println!("{}", outcome_line(&outcome));
        outcomes.push(outcome);
    }
    print_report(&Report::new(&outcomes), pricing::price_of(&model));
    Ok(())
}

/// One case is one trace, whose root shows the message and the label it got.
#[tracing::instrument(
    name = "safety-eval",
    skip_all,
    fields(
        session.id = session_id,
        input.value = case.message.as_str(),
        output.value = tracing::field::Empty,
    )
)]
async fn traced_classify(
    classifier: &Classifier,
    client: &dyn LlmClient,
    case: &Case,
    session_id: &str,
) -> Result<Outcome> {
    let outcome = classifier.classify(client, case).await?;
    let output = match &outcome.predicted {
        Ok(label) => label_name(*label),
        Err(_) => "no label",
    };
    tracing::Span::current().record("output.value", output);
    Ok(outcome)
}

fn label_name(label: SafetyLabel) -> &'static str {
    match label {
        SafetyLabel::Safe => "SAFE",
        SafetyLabel::PsychCrisis => "PSYCH_CRISIS",
        SafetyLabel::MedicalEmergency => "MEDICAL_EMERGENCY",
    }
}

fn outcome_line(outcome: &Outcome) -> String {
    let expected = label_name(outcome.expected);
    match &outcome.predicted {
        Ok(predicted) if *predicted == outcome.expected => {
            format!("✓ {}  {expected}", outcome.case_id)
        }
        Ok(predicted) => format!(
            "✗ {}  expected {expected}, got {}",
            outcome.case_id,
            label_name(*predicted)
        ),
        Err(reason) => format!(
            "! {}  expected {expected}, failed: {reason}",
            outcome.case_id
        ),
    }
}

fn percent(share: Option<f64>) -> String {
    share.map_or_else(
        || "n/a".to_owned(),
        |share| format!("{:.0}%", share * 100.0),
    )
}

fn print_report(report: &Report, price: Option<pricing::Price>) {
    println!("\nexpected ↓ / predicted →");
    print!("{:<18}", "");
    for predicted in LABELS {
        print!("{:>18}", label_name(predicted));
    }
    println!();
    for expected in LABELS {
        print!("{:<18}", label_name(expected));
        for predicted in LABELS {
            print!("{:>18}", report.count(expected, predicted));
        }
        println!();
    }
    println!(
        "\naccuracy {} · failures {}",
        percent(report.accuracy()),
        report.failures()
    );
    for label in [SafetyLabel::PsychCrisis, SafetyLabel::MedicalEmergency] {
        println!(
            "{:<18} recall {} · precision {}",
            label_name(label),
            percent(report.recall(label)),
            percent(report.precision(label))
        );
    }
    if let (Some(median), Some(max)) = (report.median_latency(), report.max_latency()) {
        println!(
            "latency on Bedrock: median {} ms, max {} ms",
            median.as_millis(),
            max.as_millis()
        );
    }
    if let Some(price) = price {
        println!("cost ${:.4}", report.cost(&price));
    }
}

/// One message of the dataset, with the label decided by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    pub id: String,
    pub message: String,
    pub expected: SafetyLabel,
}

/// A line of the dataset, in the shape of a Langfuse dataset item.
#[derive(Deserialize)]
struct Line {
    id: String,
    input: Input,
    expected_output: Expected,
}

#[derive(Deserialize)]
struct Input {
    message: String,
}

#[derive(Deserialize)]
struct Expected {
    label: SafetyLabel,
}

/// The cases of a JSONL dataset, one per non-empty line.
pub fn read_cases(text: &str) -> Result<Vec<Case>> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let line: Line = serde_json::from_str(line)
                .with_context(|| format!("line {} of the dataset", index + 1))?;
            Ok(Case {
                id: line.id,
                message: line.input.message,
                expected: line.expected_output.label,
            })
        })
        .collect()
}

/// How to call the model, from the hints in the prompt's `config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallSettings {
    pub role: ModelRole,
    pub max_tokens: u32,
}

impl CallSettings {
    /// A role that is missing or unknown is an error rather than a default, so that a typo in a
    /// prompt file never sends it quietly to another model.
    pub fn from_config(config: &Value) -> Result<Self> {
        let role = match config.get("model_role").and_then(Value::as_str) {
            Some("fast") => ModelRole::Fast,
            Some("main") => ModelRole::Main,
            Some(other) => bail!("the prompt's model_role {other} is neither fast nor main"),
            None => bail!("the prompt's config has no model_role"),
        };
        let max_tokens = config
            .get("max_tokens")
            .and_then(Value::as_u64)
            .context("the prompt's config has no max_tokens")?;
        Ok(Self {
            role,
            max_tokens: u32::try_from(max_tokens).context("the prompt's max_tokens is too big")?,
        })
    }
}

/// The safety classifier, ready to call: the prompt compiled into the binary and the model its
/// role points to.
pub struct Classifier {
    pub prompt: Prompt,
    pub model: ModelId,
    pub max_tokens: u32,
}

/// What the classifier did with one case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub case_id: String,
    pub expected: SafetyLabel,
    /// The label, or why the reply gave none. A failure is not a wrong label: it says nothing
    /// about the prompt, so it stays out of the quality numbers.
    pub predicted: Result<SafetyLabel, String>,
    pub usage: Usage,
    /// The latency Bedrock reports.
    pub latency: Duration,
}

impl Classifier {
    /// Classifies one case. An error from the client stops the run: it is a configuration or a
    /// service problem, which no further case would avoid.
    pub async fn classify(&self, client: &dyn LlmClient, case: &Case) -> Result<Outcome> {
        let rendered = self.prompt.render(&[("message", &case.message)])?;
        let request = LlmRequest {
            model: self.model.clone(),
            system: rendered.system,
            messages: rendered.messages,
            max_tokens: self.max_tokens,
            extra: Value::Null,
            prompt: Some(rendered.prompt),
            output_schema: Some(SafetyVerdict::output_schema()),
        };
        let response = client
            .complete(request)
            .await
            .with_context(|| format!("classifying {}", case.id))?;
        let reply = text_of(&response.message);
        let predicted = match response.stop_reason {
            StopReason::EndTurn => serde_json::from_str::<SafetyVerdict>(&reply)
                .map(|verdict| verdict.label)
                .map_err(|err| format!("not a verdict ({err}): {reply}")),
            other => Err(format!("stopped on {other:?}: {reply}")),
        };
        Ok(Outcome {
            case_id: case.id.clone(),
            expected: case.expected,
            predicted,
            usage: response.usage,
            latency: response.latency,
        })
    }
}

/// How the classifier did over a run. Failures are counted apart: the quality numbers are over
/// the cases that got a label.
pub struct Report<'a> {
    outcomes: &'a [Outcome],
}

impl<'a> Report<'a> {
    pub fn new(outcomes: &'a [Outcome]) -> Self {
        Self { outcomes }
    }

    /// The labelled cases, as pairs of expected and predicted label.
    fn labelled(&self) -> impl Iterator<Item = (SafetyLabel, SafetyLabel)> + '_ {
        self.outcomes.iter().filter_map(|outcome| {
            outcome
                .predicted
                .as_ref()
                .ok()
                .map(|predicted| (outcome.expected, *predicted))
        })
    }

    /// One cell of the confusion matrix.
    pub fn count(&self, expected: SafetyLabel, predicted: SafetyLabel) -> usize {
        self.labelled()
            .filter(|pair| *pair == (expected, predicted))
            .count()
    }

    pub fn failures(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|outcome| outcome.predicted.is_err())
            .count()
    }

    /// The share of labelled cases whose label is the expected one.
    pub fn accuracy(&self) -> Option<f64> {
        share(self.labelled(), |(expected, predicted)| {
            expected == predicted
        })
    }

    /// Of the real cases of a label, the share the classifier caught: read along a row.
    pub fn recall(&self, label: SafetyLabel) -> Option<f64> {
        share(
            self.labelled().filter(|(expected, _)| *expected == label),
            |(_, predicted)| predicted == label,
        )
    }

    /// Of the alarms for a label, the share that were real: read down a column.
    pub fn precision(&self, label: SafetyLabel) -> Option<f64> {
        share(
            self.labelled().filter(|(_, predicted)| *predicted == label),
            |(expected, _)| expected == label,
        )
    }

    pub fn median_latency(&self) -> Option<Duration> {
        let mut latencies: Vec<_> = self.outcomes.iter().map(|o| o.latency).collect();
        latencies.sort();
        latencies.get(latencies.len() / 2).copied()
    }

    pub fn max_latency(&self) -> Option<Duration> {
        self.outcomes.iter().map(|o| o.latency).max()
    }

    /// Every call is paid for, including those that gave no label.
    pub fn cost(&self, price: &pricing::Price) -> f64 {
        self.outcomes
            .iter()
            .map(|outcome| pricing::cost(&outcome.usage, price))
            .sum()
    }
}

/// The share of items that pass, or `None` when there are none.
fn share<T>(items: impl Iterator<Item = T>, passes: impl Fn(T) -> bool) -> Option<f64> {
    let (mut total, mut passed) = (0_u32, 0_u32);
    for item in items {
        total += 1;
        if passes(item) {
            passed += 1;
        }
    }
    (total > 0).then(|| f64::from(passed) / f64::from(total))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use futures::stream::BoxStream;
    use llm_core::{
        ContentBlock, LlmError, LlmEvent, LlmRequest, LlmResponse, Message, PromptRef, Role,
        StopReason, Usage,
    };
    use prompts::{SafetyVerdict, safety_classifier};

    use super::*;

    fn safety_v1() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../datasets/safety/v1.jsonl"
        ))
        .unwrap()
    }

    #[test]
    fn the_safety_dataset_reads_as_cases_with_one_of_the_three_labels() {
        let cases = read_cases(&safety_v1()).unwrap();

        let count = |label| cases.iter().filter(|case| case.expected == label).count();
        assert_eq!(cases.len(), 40);
        assert_eq!(count(SafetyLabel::Safe), 15);
        assert_eq!(count(SafetyLabel::PsychCrisis), 14);
        assert_eq!(count(SafetyLabel::MedicalEmergency), 11);
        assert_eq!(cases[0].id, "safety-001");
        assert_eq!(
            cases[0].message,
            "Vorrei prenotare una visita dermatologica per un neo che è cambiato un po'."
        );
    }

    #[test]
    fn a_case_with_an_unknown_label_is_an_error_that_names_its_line() {
        let text = concat!(
            r#"{"id": "a", "input": {"message": "Ciao"}, "expected_output": {"label": "SAFE"}}"#,
            "\n",
            r#"{"id": "b", "input": {"message": "Ciao"}, "expected_output": {"label": "URGENT"}}"#,
            "\n",
        );

        let err = read_cases(text).unwrap_err();

        assert!(err.to_string().contains("line 2"), "{err}");
    }

    #[test]
    fn the_safety_classifier_runs_on_the_fast_model_with_its_own_ceiling() {
        let config = serde_json::json!({ "model_role": "fast", "max_tokens": 1024 });

        let settings = CallSettings::from_config(&config).unwrap();

        assert_eq!(
            settings,
            CallSettings {
                role: ModelRole::Fast,
                max_tokens: 1024
            }
        );
    }

    #[test]
    fn a_main_role_runs_on_the_main_model() {
        let config = serde_json::json!({ "model_role": "main", "max_tokens": 200 });

        assert_eq!(
            CallSettings::from_config(&config).unwrap().role,
            ModelRole::Main
        );
    }

    #[test]
    fn a_role_that_is_unknown_or_missing_is_an_error() {
        let unknown = serde_json::json!({ "model_role": "huge", "max_tokens": 200 });
        let missing = serde_json::json!({ "max_tokens": 200 });

        assert!(CallSettings::from_config(&unknown).is_err());
        assert!(CallSettings::from_config(&missing).is_err());
    }

    /// Replies with a fixed text and remembers the request it received.
    struct FakeClient {
        reply: &'static str,
        stop_reason: StopReason,
        received: Mutex<Option<LlmRequest>>,
    }

    impl FakeClient {
        fn replying(reply: &'static str, stop_reason: StopReason) -> Self {
            Self {
                reply,
                stop_reason,
                received: Mutex::new(None),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for FakeClient {
        fn prepare(&self, req: LlmRequest) -> LlmRequest {
            req
        }

        async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
            *self.received.lock().unwrap() = Some(req);
            Ok(LlmResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text(self.reply.into())],
                },
                stop_reason: self.stop_reason.clone(),
                usage: Usage {
                    input_tokens: 600,
                    output_tokens: 30,
                    ..Usage::default()
                },
                latency: Duration::from_millis(700),
            })
        }

        async fn stream(
            &self,
            _req: LlmRequest,
        ) -> Result<BoxStream<'static, Result<LlmEvent, LlmError>>, LlmError> {
            unimplemented!("the runner does not stream")
        }
    }

    fn classifier() -> Classifier {
        Classifier {
            prompt: safety_classifier().unwrap(),
            model: ModelId::new("eu.test-model"),
            max_tokens: 1024,
        }
    }

    fn case() -> Case {
        Case {
            id: "safety-031".into(),
            message: "Ho un dolore forte al petto.".into(),
            expected: SafetyLabel::MedicalEmergency,
        }
    }

    #[tokio::test]
    async fn a_case_is_sent_as_the_rendered_prompt_with_its_schema_model_and_version() {
        let client = FakeClient::replying(
            r#"{"label": "MEDICAL_EMERGENCY", "reason": "Chest pain."}"#,
            StopReason::EndTurn,
        );

        classifier().classify(&client, &case()).await.unwrap();

        let sent = client.received.lock().unwrap().take().unwrap();
        let rendered = safety_classifier()
            .unwrap()
            .render(&[("message", "Ho un dolore forte al petto.")])
            .unwrap();
        assert_eq!(
            sent,
            LlmRequest {
                model: ModelId::new("eu.test-model"),
                system: rendered.system,
                messages: rendered.messages,
                max_tokens: 1024,
                extra: Value::Null,
                prompt: Some(PromptRef {
                    name: "safety-classifier".into(),
                    version: 4,
                }),
                output_schema: Some(SafetyVerdict::output_schema()),
            }
        );
    }

    #[tokio::test]
    async fn the_outcome_holds_the_predicted_label_its_tokens_and_latency() {
        let client = FakeClient::replying(
            r#"{"label": "PSYCH_CRISIS", "reason": "Distress."}"#,
            StopReason::EndTurn,
        );

        let outcome = classifier().classify(&client, &case()).await.unwrap();

        assert_eq!(
            outcome,
            Outcome {
                case_id: "safety-031".into(),
                expected: SafetyLabel::MedicalEmergency,
                predicted: Ok(SafetyLabel::PsychCrisis),
                usage: Usage {
                    input_tokens: 600,
                    output_tokens: 30,
                    ..Usage::default()
                },
                latency: Duration::from_millis(700),
            }
        );
    }

    #[tokio::test]
    async fn a_reply_that_is_not_a_verdict_is_a_failure_not_a_label() {
        let client = FakeClient::replying("I think this is an emergency.", StopReason::EndTurn);

        let outcome = classifier().classify(&client, &case()).await.unwrap();

        assert!(outcome.predicted.is_err(), "{outcome:?}");
    }

    #[tokio::test]
    async fn a_reply_cut_off_by_max_tokens_is_a_failure_even_when_it_parses() {
        let client = FakeClient::replying(
            r#"{"label": "SAFE", "reason": "Fine."}"#,
            StopReason::MaxTokens,
        );

        let outcome = classifier().classify(&client, &case()).await.unwrap();

        assert!(outcome.predicted.is_err(), "{outcome:?}");
    }

    fn outcome(expected: SafetyLabel, predicted: Result<SafetyLabel, String>) -> Outcome {
        Outcome {
            case_id: "case".into(),
            expected,
            predicted,
            usage: Usage::default(),
            latency: Duration::from_millis(500),
        }
    }

    fn repeat(n: usize, expected: SafetyLabel, predicted: SafetyLabel) -> Vec<Outcome> {
        vec![outcome(expected, Ok(predicted)); n]
    }

    /// The example of 6.10 in the tutorial: 6 real crises among 40 messages, 10 alarms, 4 of
    /// them on real crises.
    fn tutorial_example() -> Vec<Outcome> {
        use SafetyLabel::{PsychCrisis, Safe};
        [
            repeat(4, PsychCrisis, PsychCrisis),
            repeat(2, PsychCrisis, Safe),
            repeat(6, Safe, PsychCrisis),
            repeat(28, Safe, Safe),
        ]
        .concat()
    }

    #[test]
    fn the_report_counts_each_pair_of_expected_and_predicted_labels() {
        let outcomes = tutorial_example();
        let report = Report::new(&outcomes);

        assert_eq!(report.count(SafetyLabel::PsychCrisis, SafetyLabel::Safe), 2);
        assert_eq!(report.count(SafetyLabel::Safe, SafetyLabel::PsychCrisis), 6);
        assert_eq!(
            report.count(SafetyLabel::MedicalEmergency, SafetyLabel::Safe),
            0
        );
    }

    #[test]
    fn recall_and_precision_match_the_tutorial_example() {
        let outcomes = tutorial_example();
        let report = Report::new(&outcomes);

        assert_eq!(report.recall(SafetyLabel::PsychCrisis), Some(4.0 / 6.0));
        assert_eq!(report.precision(SafetyLabel::PsychCrisis), Some(4.0 / 10.0));
        assert_eq!(report.accuracy(), Some(32.0 / 40.0));
    }

    #[test]
    fn a_class_with_no_real_cases_or_no_alarms_has_no_recall_or_precision() {
        let outcomes = tutorial_example();
        let report = Report::new(&outcomes);

        assert_eq!(report.recall(SafetyLabel::MedicalEmergency), None);
        assert_eq!(report.precision(SafetyLabel::MedicalEmergency), None);
    }

    #[test]
    fn failures_are_counted_apart_and_stay_out_of_the_quality_numbers() {
        let mut outcomes = tutorial_example();
        outcomes.push(outcome(
            SafetyLabel::PsychCrisis,
            Err("not a verdict".into()),
        ));
        let report = Report::new(&outcomes);

        assert_eq!(report.failures(), 1);
        assert_eq!(report.recall(SafetyLabel::PsychCrisis), Some(4.0 / 6.0));
        assert_eq!(report.accuracy(), Some(32.0 / 40.0));
    }

    #[test]
    fn the_report_gives_the_median_and_the_slowest_latency() {
        let outcomes: Vec<_> = [700, 100, 300]
            .into_iter()
            .map(|ms| Outcome {
                latency: Duration::from_millis(ms),
                ..outcome(SafetyLabel::Safe, Ok(SafetyLabel::Safe))
            })
            .collect();
        let report = Report::new(&outcomes);

        assert_eq!(report.median_latency(), Some(Duration::from_millis(300)));
        assert_eq!(report.max_latency(), Some(Duration::from_millis(700)));
    }

    #[test]
    fn the_cost_adds_up_every_call_failures_included() {
        let tokens = Usage {
            input_tokens: 600,
            output_tokens: 30,
            ..Usage::default()
        };
        let outcomes = vec![
            Outcome {
                usage: tokens,
                ..outcome(SafetyLabel::Safe, Ok(SafetyLabel::Safe))
            },
            Outcome {
                usage: tokens,
                ..outcome(SafetyLabel::Safe, Err("not a verdict".into()))
            },
        ];
        let price = pricing::Price {
            input: 1.10,
            output: 5.50,
            cache_write: 1.375,
            cache_read: 0.11,
        };

        // 600 × 1.10 / 1M + 30 × 5.50 / 1M = 0.000825 per call.
        let cost = Report::new(&outcomes).cost(&price);

        assert!((cost - 0.00165).abs() < 1e-12, "{cost}");
    }
}
