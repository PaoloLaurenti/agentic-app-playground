# Formatting and lints: the same checks CI runs.
check:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings

# The test suite, run by nextest.
test:
    cargo nextest run

# The CLI, with arguments passed through: `just run hello --times 5`.
run *args:
    cargo run -q -p app -- {{args}}
