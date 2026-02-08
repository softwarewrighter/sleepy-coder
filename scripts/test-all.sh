#!/bin/bash
# Run all tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
RUST_DIR="$PROJECT_DIR/rust"

echo "=== Running Rust tests ==="
cd "$RUST_DIR"
cargo test

echo ""
echo "=== All tests passed ==="
