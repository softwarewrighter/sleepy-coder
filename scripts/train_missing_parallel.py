#!/usr/bin/env python3
"""Train only missing patterns in parallel."""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sft" / "patterns"
ADAPTERS_DIR = PROJECT_ROOT / "runs" / "adapters" / "pattern_adapters"


def get_missing_patterns():
    """Find patterns that don't have trained adapters."""
    all_patterns = {p.stem for p in DATA_DIR.glob("*.jsonl")}
    trained = {p.name for p in ADAPTERS_DIR.iterdir() if p.is_dir()} if ADAPTERS_DIR.exists() else set()
    return sorted(all_patterns - trained)


def train_one(pattern):
    """Train a single adapter. Returns (pattern, success, path)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ADAPTERS_DIR / pattern / timestamp
    data_file = DATA_DIR / f"{pattern}.jsonl"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "cuda" / "scripts" / "train.py"),
        "--data", str(data_file),
        "--steps", "30",
        "--output", str(output_dir),
        "--lr", "1e-4",
        "--lora-r", "8",
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT / "cuda", capture_output=True)
    success = result.returncode == 0 and (output_dir / "adapter").exists()
    return pattern, success, output_dir / "adapter" if success else None


def main():
    missing = get_missing_patterns()
    print(f"Missing patterns: {len(missing)}")
    for p in missing:
        print(f"  - {p}")
    print()

    if not missing:
        print("All patterns trained!")
        return

    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Training with {max_workers} parallel workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_one, p): p for p in missing}
        for i, future in enumerate(as_completed(futures), 1):
            pattern, success, path = future.result()
            status = "OK" if success else "FAILED"
            print(f"[{i}/{len(missing)}] {pattern}: {status}")
            results.append((pattern, success, path))

    success_count = sum(1 for _, s, _ in results if s)
    print(f"\nDone: {success_count}/{len(missing)} successful")

    # List all adapters now
    print(f"\nTotal adapters: {len(list(ADAPTERS_DIR.iterdir()))}")


if __name__ == "__main__":
    main()
