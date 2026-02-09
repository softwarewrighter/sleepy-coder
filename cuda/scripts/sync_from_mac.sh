#!/bin/bash
# Sync training data and checkpoints from Mac to Linux
#
# Run this ON THE LINUX BOX after cloning the repo
#
# Usage:
#   ./sync_from_mac.sh user@mac-hostname
#
# What gets synced:
#   - Training data: data/sft/train.jsonl
#   - Checkpoints: runs/adapters/ (if continuing training)
#   - Episodes: rust/data/episodes/ (for eval metrics)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 user@mac-hostname"
    echo "Example: $0 mike@mikes-macbook.local"
    exit 1
fi

MAC_HOST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Sync from Mac to Linux ==="
echo "Mac host: $MAC_HOST"
echo "Repo root: $REPO_ROOT"
echo ""

# Remote repo path (usually same as local)
REMOTE_REPO="/Users/mike/github/softwarewrighter/sleepy-coder"

# Create local directories
mkdir -p "$REPO_ROOT/data/sft"
mkdir -p "$REPO_ROOT/runs/adapters"
mkdir -p "$REPO_ROOT/rust/data/episodes"

echo "=== Syncing training data ==="
rsync -avz --progress "$MAC_HOST:$REMOTE_REPO/data/sft/" "$REPO_ROOT/data/sft/"

echo ""
echo "=== Syncing episodes ==="
rsync -avz --progress "$MAC_HOST:$REMOTE_REPO/rust/data/episodes/" "$REPO_ROOT/rust/data/episodes/"

echo ""
echo "=== Syncing adapters (if any) ==="
rsync -avz --progress "$MAC_HOST:$REMOTE_REPO/runs/adapters/" "$REPO_ROOT/runs/adapters/" 2>/dev/null || echo "No adapters to sync yet"

echo ""
echo "=== Sync complete ==="
echo ""
echo "Training data: $(wc -l < "$REPO_ROOT/data/sft/train.jsonl" 2>/dev/null || echo 0) examples"
echo ""
echo "Next steps:"
echo "  cd $REPO_ROOT/cuda"
echo "  source .venv/bin/activate"
echo "  python scripts/quick_test.py"
