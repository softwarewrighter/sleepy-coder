#!/bin/bash
# Run cargo commands in the Rust workspace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/../rust"

cd "$RUST_DIR"
cargo "$@"
