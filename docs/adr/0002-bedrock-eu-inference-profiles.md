# 0002 · Amazon Bedrock in eu-west-1 with EU inference profiles

- Date: 2026-09-25
- Status: accepted

## Context

The guiding project is a healthcare assistant, so data must be processed inside the EU
(`AGENTS.md`). The router and the evaluations (Modules 10 and 11) need models from more than one
provider, and the harness is written in Rust. The AWS account is a sandbox shared with other
work, under an organization whose SCP restricts the usable regions.

## Options considered

1. Claude Platform on AWS: operated by Anthropic, with same-day features, but it offers only
   global and US inference geographies, and data may leave AWS.
2. Bedrock with `global.*` inference profiles: 10% cheaper, but calls can run outside the EU.
3. Bedrock with `eu.*` inference profiles, home region `eu-central-1` (Frankfurt).
4. Bedrock with `eu.*` inference profiles, home region `eu-south-1` (Milan).
5. Bedrock with `eu.*` inference profiles, home region `eu-west-1` (Ireland).

## Decision

Option 5. Today Bedrock is the only option that offers several LLMs with EU data residency, and
its Anthropic models are reachable only through inference profiles, so `eu.*` is the only EU
route. Frankfurt is denied by the SCP. Milan lists the same Claude 5 profiles as Ireland, but not
the Llama and Mistral ones that Module 11 needs (42 inference profiles against 50 on
2026-09-24). A dedicated `bedrock-playground` AWS profile points to Ireland, so the shared
`sandbox` profile keeps its own region.

## Consequences

Data stays in the EU, and one API (Converse) and one official Rust SDK cover every provider.
EU prices are 10% above global ones (Haiku 4.5: 1.10 and 5.50 USD per million input and output
tokens, against 1.00 and 5.00), and new Anthropic features reach Bedrock later than the
Anthropic API. An `eu.*` profile can run a call in any of its EU destination regions, so the
SCP must allow Bedrock invocations in all of them: the organization admins changed it on
2026-09-25, after every call routed to `eu-north-1` had been denied. Claude is billed as an AWS
Marketplace product, which cost filters must take into account. Revisit before production
(Module 15), in whatever account hosts it, and whenever another provider offers several LLMs
with EU data residency.
