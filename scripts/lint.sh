#!/bin/bash
# Run linting and formatting
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
RUST_DIR="$PROJECT_DIR/rust"

echo "=== Running Rust clippy ==="
cd "$RUST_DIR"
cargo clippy --all-targets --all-features -- -D warnings

echo ""
echo "=== Running Rust fmt ==="
cargo fmt --all

echo ""
echo "=== Checking Rust fmt ==="
cargo fmt --all -- --check

echo ""
echo "=== Validating markdown ==="
cd "$PROJECT_DIR"
if command -v markdown-checker &> /dev/null; then
    markdown-checker -f "docs/*.md"
else
    echo "markdown-checker not found, skipping"
fi

echo ""
echo "=== Lint complete ==="
