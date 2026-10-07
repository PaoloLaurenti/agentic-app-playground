# 0009 · Prompts live in the repository, and a commit fixes the one that runs

- Date: 2026-10-07
- Status: accepted

## Context

The tutorial's plan for step 6.5 made Langfuse the source of every prompt. The app would fetch
the version labelled `production`, keep it in a cache with a TTL, and fall back to a snapshot in
`prompts/` when Langfuse was unreachable. That plan has four costs:

1. **Two sources of truth.** The prompt lives in Langfuse and in its snapshot, and they can drift
   apart.
2. **Changes outside review.** Moving `production` in the Langfuse interface changes what the app
   does, with no pull request and no deploy.
3. **A runtime dependency.** Every start, and every cache expiry, calls an external service.
4. **No reproducibility.** The same commit can run different prompts, depending on where the
   label points at that moment.

All four come from one feature: changing the prompt the app runs without a deploy. In a
healthcare domain, where even the escalation replies are fixed texts decided by people (9.4),
that feature is not wanted.

## Options considered

1. Langfuse as the source, served by label, with a cache and a fallback file. Protected labels
   and snapshot tests soften costs 1 to 3, but a label moved in the interface still changes the
   running app.
2. The repository as the source. Prompts are compiled into the binary, and `just prompts-push`
   copies new versions to Langfuse.
3. The repository as the source, with the app fetching from Langfuse the exact version the
   repository names. This makes runs reproducible but keeps the runtime dependency, to fetch a
   text the repository already holds.

## Decision

Option 2. Each prompt is a file in `prompts/`, holding its messages, its `config` and the version
number Langfuse gave it, embedded in the binary with `include_str!`. A commit fixes exactly which
prompt runs, and changing it takes a pull request and a deploy. `just prompts-push` creates a
new version in Langfuse when a file's text differs from the latest one there, and writes the
version number back into the file. Langfuse keeps every version and links it to the generations
that used it (6.6), but it never decides which one runs. Its labels mean nothing to the app.

## Consequences

The prompts crate needs no download, no cache and no fallback, so its tests need no network.
There is no prompt rollback without a deploy, nobody can edit a prompt from the Langfuse
interface, and a prompt canary is an app canary (15.8). Iterating (6.11) means editing the file,
pushing it and measuring. The Playground stays useful for trying a variant before it is written
to the file. A file whose text no longer matches the version it names would link generations to
the wrong version: a contract test compares the two. The Module 6 title keeps Langfuse Prompt
Management, which now serves as the registry of versions rather than as their source.
