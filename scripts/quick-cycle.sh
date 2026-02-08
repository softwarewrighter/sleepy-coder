#!/bin/bash
# Run a quick demo cycle: day -> sleep -> eval -> plot
# Usage: ./scripts/quick-cycle.sh [N_CYCLES] [N_TASKS] [N_TRAIN_STEPS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
RUST_DIR="$PROJECT_DIR/rust"
PY_DIR="$PROJECT_DIR/py"

N_CYCLES="${1:-6}"
N_TASKS="${2:-10}"
N_STEPS="${3:-300}"

echo "=== Sleepy Coder Quick Cycle Demo ==="
echo "Cycles: $N_CYCLES, Tasks/cycle: $N_TASKS, Train steps: $N_STEPS"
echo ""

for i in $(seq 1 "$N_CYCLES"); do
    echo "=== Cycle $i/$N_CYCLES ==="

    echo "  [DAY] Running agent on $N_TASKS tasks..."
    # TODO: cd "$RUST_DIR" && cargo run -p sleepy-coder -- run-day --n "$N_TASKS"

    echo "  [SLEEP] Training on captured episodes..."
    # TODO: cd "$PY_DIR" && python -m sleepy_pact.train.lora_train --max-steps "$N_STEPS"

    echo "  [EVAL] Evaluating model..."
    # TODO: cd "$RUST_DIR" && cargo run -p sleepy-coder -- eval

    echo "  [PLOT] Generating visualizations..."
    # TODO: cd "$PY_DIR" && python -m sleepy_pact.viz.plots

    echo ""
done

echo "=== Demo complete ==="
echo "Check runs/latest/viz/ for plots"
