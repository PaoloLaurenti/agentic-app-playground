# 0011 · `SAFETY` runs on the fast model

- Date: 2026-10-11
- Status: accepted

## Context

`SAFETY` classifies every user message before anything else runs, so its cost and latency are
paid on every turn. Step 6.12 asks whether the main model classifies better than the fast one.
EXP-004 measured `safety-classifier` v4 on Claude Haiku 4.5 (`fast`), EXP-008 the same prompt
on Claude Sonnet 4.6 (`main`), both over `datasets/safety/v1.jsonl`, 40 synthetic cases.

## Options considered

1. Claude Haiku 4.5, the `fast` role: 1.10 and 5.50 USD per million input and output tokens.
2. Claude Sonnet 4.6, the `main` role: 3.30 and 16.50 USD per million tokens.

## Decision

Option 1. Both models give the same label to every case, 39 of 40, with the same single error,
`safety-028`, which is a clinical question rather than a limit of the model. Sonnet costs 2.9
times as much (0.119 against 0.041 USD for the 40 cases), and its median Bedrock latency, 1,290
ms, is within the 1,339 to 1,728 ms that Haiku showed over five runs. The prompt's `config`
keeps `model_role = "fast"`.

## Consequences

The decision rests on a dataset where both models are near the ceiling: 40 cases cannot show a
small advantage, and the labels have not been validated by a clinician. Revisit it when a
harder v2 of the dataset exists, when the `fast` role moves to another model, or when Module 10
brings routing, which could send only uncertain messages to the main model.
