#!/usr/bin/env python3
"""Direct Share consolidation runner."""
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from share_proper import ShareAlgorithm
import torch

ADAPTERS_FILE = PROJECT_ROOT / "runs" / "adapters" / "pattern_adapters" / "adapters_for_share.txt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "share" / "pattern_share_51"


def main():
    # Load adapter paths
    with open(ADAPTERS_FILE) as f:
        adapters = [Path(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(adapters)} adapter paths")

    # Run Share
    share = ShareAlgorithm()
    print(f"\n=== Consolidating {len(adapters)} adapters ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        share.consolidate(adapters, OUTPUT_DIR, target_variance=0.6)
        print(f"\nShare output: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
