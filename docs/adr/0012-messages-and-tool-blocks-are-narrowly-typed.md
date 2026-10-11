# 0012 · Messages and tool blocks are narrowly typed

- Date: 2026-10-11
- Status: accepted

## Context

Step 7.2 teaches `llm-core` about tools: the model asks for one in its turn, and the result goes
back in the next user turn. Until now a message was `Message { role, content: Vec<ContentBlock> }`,
where every block could sit in either role, and the reply was a `Message` with a role that is
always the assistant's. Adding tool blocks to that shape would let the code write a tool request
in a user turn, or a tool result in the model's, which Converse rejects only at call time. The
rule in `AGENTS.md` asks for the narrowest type that fits, so that such states cannot be written.

## Options considered

1. Keep one `ContentBlock` for both roles and add `ToolUse` and `ToolResult` to it, as the
   tutorial first sketched, with `is_error: bool` on the result and `String` for ids, names,
   inputs and schemas.
2. One type of block per role, and a narrow type for each part of a tool block.

## Decision

Option 2:

- `Message` is an enum, `User(Vec<UserBlock>)` or `Assistant(Vec<AssistantBlock>)`, and `Role`
  disappears. `UserBlock` carries text, the cache point and, from step 7.2, a tool result;
  `AssistantBlock` carries text and a tool request. `LlmResponse.content` is a
  `Vec<AssistantBlock>`, with no role. The cache point stays on the user's side: the cached
  prefix ends with the last message sent, which is always the user's.
- A tool result carries a `ToolOutcome`, `Success(String)` or `Failure(String)`, rather than a
  text and a boolean: the text is the result or the reason for the failure, never both.
- `ToolName` is a newtype checked when it is built: 1 to 64 letters, digits, `_` or `-`, the
  limits of Converse's `ToolSpecification`. A wrong name fails where the tool is defined, not at
  the first call to Bedrock.
- `ToolUseId` is an opaque newtype, not checked: the provider creates it and the harness only
  copies it into the result, so it only has to be impossible to mix up with a name.
- A tool's input is a `serde_json::Map`, a JSON object, because its shape depends on the tool:
  the registry in `harness` narrows it to each tool's own `Input` type. The schema is an
  `InputSchema` newtype checked to be an object whose root has `"type": "object"`, which is all
  `llm-core` can know about it; modelling all of JSON Schema would redo what `schemars` does.

## Consequences

The compiler rejects a tool block in the wrong turn, a result that is both a success and an
error, and an id passed where a name is expected. Code that only reads the text of a reply,
such as `app` and the Langfuse generation, matches on the blocks of one role. Two limits remain
unchecked: a message may have no blocks, which Converse rejects, and an input is not checked
against its schema until the registry deserializes it.
