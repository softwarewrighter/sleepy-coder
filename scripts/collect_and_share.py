#!/usr/bin/env python3
"""Collect one adapter per pattern and run Share consolidation."""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "runs" / "adapters" / "pattern_adapters"


def collect_adapters():
    """Find one valid adapter per pattern."""
    adapters = {}

    for pattern_dir in sorted(ADAPTERS_DIR.iterdir()):
        if not pattern_dir.is_dir():
            continue

        pattern = pattern_dir.name

        # Find all adapter_config.json files
        adapter_paths = list(pattern_dir.rglob("adapter/adapter_config.json"))

        if adapter_paths:
            # Get the most recent one
            adapter_path = sorted(adapter_paths)[-1].parent
            adapters[pattern] = adapter_path

    return adapters


def main():
    adapters = collect_adapters()
    print(f"Found {len(adapters)} adapters for {len(adapters)} patterns")

    for pattern, path in sorted(adapters.items()):
        print(f"  {pattern}: {path}")

    if len(adapters) < 2:
        print("Need at least 2 adapters for Share")
        return

    # Save adapter list
    manifest = PROJECT_ROOT / "runs" / "adapters" / "pattern_adapters" / "adapters_for_share.txt"
    with open(manifest, "w") as f:
        for path in adapters.values():
            f.write(str(path) + "\n")
    print(f"\nAdapter list saved to: {manifest}")

    # Run Share consolidation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    share_output = PROJECT_ROOT / "runs" / "share" / f"pattern_share_{timestamp}"

    print(f"\n=== Running Share Consolidation ===")
    print(f"Output: {share_output}")

    adapter_paths = [str(p) for p in adapters.values()]

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "share_proper.py"),
        "consolidate",
        "--adapters", *adapter_paths,
        "-o", str(share_output),
        "-v", "0.6",  # 60% explained variance
    ]

    print(f"Running: {' '.join(cmd[:10])}... (51 adapters)")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"\nShare consolidation complete: {share_output}")
    else:
        print(f"\nShare consolidation FAILED (exit code {result.returncode})")


if __name__ == "__main__":
    main()
