# Default recipe to display help
default:
  @just --list

# Format all code
format:
  rumdl fmt .
  cargo sort -w -g
  cargo +nightly fmt --all

# Auto-fix linting issues
fix:
  rumdl check --fix .
  RUSTC_WRAPPER= cargo +nightly clippy --fix --all --allow-dirty

# Run all lints
lint:
  typos
  rumdl check .
  cargo sort -w -g -c
  cargo +nightly fmt --all -- --check
  RUSTC_WRAPPER= cargo +nightly clippy --all -- -D warnings
  cargo shear

# Run tests
test:
  cargo test --all-features

# Run BDD scenarios
bdd:
  cargo test -p ltk-core --test checkout-bdd
  cargo test -p ltk-core --test pipeline-bdd

# Run both TDD and BDD suites
test-all:
  cargo test --all-features
  cargo test -p ltk-core --test checkout-bdd
  cargo test -p ltk-core --test pipeline-bdd

# Run tests with coverage
test-coverage:
  cargo tarpaulin --all-features --workspace --timeout 300

# Build entire workspace
build:
  cargo build --workspace

# Check all targets compile
check:
  cargo check --all-targets --all-features

# Publish ltk-core to crates.io (dry run)
publish-check:
  cargo publish -p ltk-core --dry-run --allow-dirty --registry crates-io

# Publish ltk-core to crates.io
publish:
  cargo publish -p ltk-core --registry crates-io

# Check for Chinese characters
check-cn:
  rg --line-number --column "\p{Han}"

# Full CI check
ci: lint test-all build

# ============================================================
# Maintenance & Tools
# ============================================================

# Clean build artifacts
clean:
  cargo clean

# Install all required development tools
setup:
  cargo install cargo-shear
  cargo install cargo-sort
  cargo install typos-cli

# Generate documentation for the workspace
docs:
  cargo doc --no-deps --open
