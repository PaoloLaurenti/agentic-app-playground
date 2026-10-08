# Formatting and lints: the same checks CI runs.
check:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings

# The test suite, run by nextest.
test:
    cargo nextest run

# The CLI, with arguments passed through: `just run hello --times 5`. Positional arguments keep
# a quoted value such as `--message "Hello there"` in one piece.
[positional-arguments]
run *args:
    cargo run -q -p app -- "$@"

# Copies to Langfuse every prompt file whose text is not yet the version it names, and writes the
# new version number back into the file (ADR 0009).
prompts-push:
    cargo run -q -p app -- prompts-push
