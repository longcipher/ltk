# ltk (Less Token) Agent Instructions

## Non-negotiables

- Surgical edits only: every changed line must trace to the request. No drive-by refactors, no reformatting untouched code, no deleting pre-existing dead code (mention it instead).
- Fix root causes. Never silence a warning, weaken a check, or patch a symptom to make a gate pass.
- Never report completion while any `just` gate below fails. Report the failure verbatim and stop.
- Stop and ask when a request is ambiguous, conflicts with existing conventions, or requires credentials, publishing, or destructive operations.
- When a correction reveals a missing rule, add one specific rule here ("always X for Y", never "be careful with Y"); delete rules that never catch real errors.

## Scope

- ltk is a cascade log-compression pipeline for LLM/agent contexts. It cuts token usage 50-95% on repetitive log streams.
- The pipeline is four stages: `Normalizer` (strip ANSI, mask IP/UUID/timestamp/hex/path/number/port into stable placeholders) -> `Clusterer` (Jaccard-similarity online clustering with move-to-front LRU and frozen exemplars) -> optional `SemanticCompressor` (LLMLingua-style neural token pruning) -> `Formatter`.
- `crates/ltk-core` is the library (normalizer, clusterer, formatter, pipeline, onnx, rpc); `bin/ltk-cli` is the `ltk` binary (clap).
- Feature flags on `ltk-core`: `tiktoken` (OpenAI token counting), `llmlingua-onnx` (local ONNX Runtime pruning), `llmlingua-rpc` (remote LLMLingua daemon). `ltk-cli` adds `token-counter` and forwards the two Stage 2 flags.
- Formatters are `compact`, `tsv`, `toon` and `json-minimal`; the CLI writes the compressed stream to stdout and `--stats` to stderr.
- The release workflow publishes `ltk-core` to crates.io only; `ltk-cli` is never published.
- `crates/ltk-core/src/checkout.rs` is leftover BDD-demo scaffolding kept as a template only. Do not extend it and do not build features on it.
- English only in code, comments, docs, and commit messages.

## Execution

- Parallelize aggressively: dispatch subagents for independent research, exploration, and implementation.
- NEVER run two `cargo` commands concurrently, and never overlap file edits with a running `cargo` command (`target/` file lock): finish edits, wait, then run.
- NEVER use `git worktree`. Edit the current branch in place.
- TDD for behavior changes: failing crate-local test first, smallest change to pass, then run `just mutation` and fix surviving mutants.

## Verification loop

After each feature or bug fix, in order; any failure means the task is not done:

```bash
just format
just lint
just test
just mutation
```

`just lint` is exactly the tool set CI installs: `typos`, `rumdl`, `cargo sort -w -g -c`, `cargo +nightly fmt --check`, `cargo +nightly clippy --all -- -D warnings`, `cargo shear`. `just test-all` additionally covers the optional feature combinations.

## Workspace rules

- Never hand-type dependency versions. Add per-crate deps with `cargo add <crate> -p <crate>`. Shared deps go into root `[workspace.dependencies]` (numeric versions only, no default features) and are referenced with `dep = { workspace = true }`.
- Sub-crates inherit `version`/`edition`/`license`/`repository` via `workspace = true` and must set `[lints] workspace = true`.
- There is no `rust-toolchain.toml`; CI installs a nightly toolchain, so use `cargo +nightly fmt` and `cargo +nightly clippy` exactly as the Justfile does.
- `.cargo/config.toml` sets `rustc-wrapper = "kache"`, which breaks clippy's lints and `--fix`: prefix clippy with `RUSTC_WRAPPER=` (the Justfile does; CI sets `RUSTC_WRAPPER: ""` for the same reason).
- `allow_attributes` is `deny`, and the tree currently uses no `#[allow]`/`#[expect]` escape hatch at all. `unwrap_used`, `expect_used`, `panic`, `todo`, `unimplemented`, `dbg_macro`, `print_stdout` and `print_stderr` are denied as well.
- Every entry in the root `[workspace.dependencies]` table is referenced by at least one crate today. `cargo shear` must stay green, so delete an entry as soon as nothing uses it instead of leaving it reserved.

## Dependency choices

- Errors: `thiserror` in `ltk-core`, `eyre` in `ltk-cli`. Never `anyhow`.
- Observability: `tracing` in the library, `tracing-subscriber` in the CLI. Never `log`.
- Text work: `regex` for the normalizer patterns, `strip-ansi-escapes` for ANSI removal, `serde`/`serde_json` only for the RPC wire format.
- Small strings and trusted hash keys: `compact_str` and `ahash`. Do not add a general-purpose cache crate; the clusterer already owns its LRU.
- Stage 2 remote pruning uses `hpx` with the `json` feature. `hpx` is async-only, so the sync `SemanticCompressor::compress` trait method bridges to it through a dedicated tokio runtime inside `crates/ltk-core/src/rpc.rs`. Do not make the trait async.
- Stage 2 local pruning uses `ort` (ONNX Runtime) plus `tokenizers`; the `tiktoken` feature uses `tiktoken-rs`. These are heavy optional deps: keep them behind their feature flags.
- No `reqwest`, no `sqlx`, no `dashmap`, no `parking_lot` in this tree.

## Code conventions

- The pipeline is stage-based and each stage is a trait plus a default implementation: `Normalizer`, `Clusterer`, `SemanticCompressor`, `Formatter`. New behaviour belongs behind a trait so the stage can be swapped or faked in tests.
- Clustering is invariant-critical: every input line must land in exactly one cluster, and the occurrence counts printed as `[xN]` must sum to the original line count. `crates/ltk-core/tests/proptest.rs` asserts this; keep it green.
- Masking must be deterministic and reversible enough for grouping: the same raw line always normalizes to the same masked form, and the raw text is preserved in the normalized record.
- Property tests live in `crates/ltk-core/tests/proptest.rs` and are part of the ordinary `cargo test` path; do not create a separate property-test command.
- `cargo test --all-features` pulls in the ONNX and RPC stages; `just test-all` runs the default and all-features configurations. New code must compile and pass in both.
- Public items need doc comments (`missing_docs` warns), and public items need `Debug` (`missing_debug_implementations` warns).
