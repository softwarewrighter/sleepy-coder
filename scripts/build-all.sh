#!/bin/bash
# Build all Rust crates and Python package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
RUST_DIR="$PROJECT_DIR/rust"
PY_DIR="$PROJECT_DIR/py"

echo "=== Building Rust workspace ==="
cd "$RUST_DIR"
cargo build --release

echo ""
echo "=== Checking Python package ==="
cd "$PY_DIR"
if command -v uv &> /dev/null; then
    echo "Using uv to check Python deps..."
    uv pip check 2>/dev/null || echo "Python deps not installed (run: uv pip install -e .)"
elif command -v pip &> /dev/null; then
    pip check 2>/dev/null || echo "Python deps not installed (run: pip install -e .)"
else
    echo "No Python package manager found (uv or pip)"
fi

echo ""
echo "=== Build complete ==="
