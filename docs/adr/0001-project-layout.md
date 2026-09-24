# 0001 · The project layout grows with the modules

- Date: 2026-09-24
- Status: accepted

## Context

Today the project is a single crate. The layout it is heading towards, listed in `AGENTS.md`,
will have eight crates plus `prompts/` and `datasets/`. This is a learning project: a piece
created before it is needed has no visible reason to exist, and that makes the whole structure
harder to understand. The first real boundary appears in Module 4, where `llm-core` must stay
free of any AWS dependency.

## Options considered

1. Create the full workspace with every crate on day one.
2. Start with a single crate and add each crate, folder, dependency and `just` recipe in the
   module that first needs it, turning the project into a workspace when the second crate
   appears.
3. Keep a single crate for good and separate concerns with Rust modules instead of crates.

## Decision

Option 2. Every piece arrives together with the reason it exists. Option 3 was rejected because
a module cannot stop code from importing the AWS SDK, while a crate boundary can: the
dependency graph enforces it at compile time.

## Consequences

Each change stays small and understood, and there are no empty crates waiting for a purpose.
The price is one conversion in Module 4, from a single package to a virtual workspace with
`resolver = "3"`, moving `src/main.rs` into `crates/app`. Revisit this decision if a crate
boundary ever turns out to be premature, or when the workspace grows past the planned layout.
