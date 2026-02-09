#!/bin/bash
# Train LoRA adapters for all domains and consolidate with Share

set -e

cd /home/mike/github/softwarewrighter/sleepy-coder
source cuda/.venv/bin/activate

DOMAINS_DIR="data/sft/domains"
ADAPTERS_DIR="runs/adapters/share_domains"
STEPS=30  # Quick training for demo

echo "=== Training Domain Adapters ==="

# Train each domain
for domain in yew_wasm axum_server sqlx_db cli_clap refactoring style_metrics; do
    echo ""
    echo "=== Training ${domain} ==="

    mkdir -p "${ADAPTERS_DIR}/${domain}"

    python cuda/scripts/train.py \
        --data "${DOMAINS_DIR}/${domain}.jsonl" \
        --output "${ADAPTERS_DIR}/${domain}" \
        --steps ${STEPS} \
        --lr 1e-4 \
        --lora-r 8
done

echo ""
echo "=== All domain adapters trained ==="
echo ""

# List adapters
echo "Adapters:"
for domain in yew_wasm axum_server sqlx_db cli_clap refactoring style_metrics; do
    latest=$(ls -t "${ADAPTERS_DIR}/${domain}" | head -1)
    echo "  ${domain}: ${ADAPTERS_DIR}/${domain}/${latest}/adapter"
done

echo ""
echo "=== Consolidating with Share ==="

# Build adapter list
ADAPTER_LIST=""
for domain in yew_wasm axum_server sqlx_db cli_clap refactoring style_metrics; do
    latest=$(ls -t "${ADAPTERS_DIR}/${domain}" | head -1)
    ADAPTER_LIST="${ADAPTER_LIST} ${ADAPTERS_DIR}/${domain}/${latest}/adapter"
done

# Consolidate
python scripts/share_proper.py consolidate \
    -a ${ADAPTER_LIST} \
    -o runs/share_consolidated \
    --variance 0.6

echo ""
echo "=== Done ==="
echo "Share model: runs/share_consolidated"
