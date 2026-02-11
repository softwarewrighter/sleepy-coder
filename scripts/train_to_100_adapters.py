#!/usr/bin/env python3
"""
Train additional adapters to reach 100+ total adapters.

Strategy:
1. Train on 6 domains (6 adapters)
2. Train 2 adapters per distilled pattern with different data splits (16 adapters)
3. Train with different ranks for more variety (24 adapters)
4. Train on combined data with different configs (4 adapters)

Total new: ~50 adapters
"""

import subprocess
import json
from pathlib import Path
import random

BASE_DIR = Path("/home/mike/github/softwarewrighter/sleepy-coder")
CUDA_DIR = Path("/mnt/a500ga/sleepy-coder/cuda")
OUTPUT_BASE = Path("/mnt/a500ga/sleepy-coder/runs/adapters/scale100")

def run_train(data_path: str, output_name: str, steps: int = 30, lr: float = 1e-4, rank: int = 8):
    """Run training with HF_HUB_OFFLINE=1"""
    output_dir = OUTPUT_BASE / output_name
    cmd = [
        "python", str(CUDA_DIR / "scripts/train.py"),
        "--data", data_path,
        "--output", str(output_dir),
        "--steps", str(steps),
        "--lr", str(lr),
        "--lora-r", str(rank),
    ]
    env = {"HF_HUB_OFFLINE": "1"}
    print(f"\n=== Training {output_name} ===")
    print(f"  Data: {data_path}")
    print(f"  Steps: {steps}, LR: {lr}, Rank: {rank}")

    result = subprocess.run(
        cmd,
        env={**dict(__import__('os').environ), **env},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    # Find the adapter directory
    adapter_dirs = list(output_dir.glob("*/adapter"))
    if adapter_dirs:
        print(f"  SUCCESS: {adapter_dirs[0]}")
        return adapter_dirs[0]
    return None


def split_jsonl(input_file: Path, output_dir: Path, splits: int = 2):
    """Split a JSONL file into multiple files."""
    with open(input_file) as f:
        lines = f.readlines()

    random.shuffle(lines)
    chunk_size = len(lines) // splits

    output_files = []
    for i in range(splits):
        start = i * chunk_size
        end = start + chunk_size if i < splits - 1 else len(lines)
        output_file = output_dir / f"{input_file.stem}_split{i+1}.jsonl"
        with open(output_file, "w") as f:
            f.writelines(lines[start:end])
        output_files.append(output_file)
        print(f"  Split {i+1}: {end - start} examples -> {output_file.name}")

    return output_files


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    trained_adapters = []

    # 1. Train on domains (6 adapters)
    print("\n" + "="*60)
    print("PHASE 1: Training on domains (6 adapters)")
    print("="*60)

    domains = ["axum_server", "cli_clap", "refactoring", "sqlx_db", "style_metrics", "yew_wasm"]
    for domain in domains:
        data_path = BASE_DIR / f"data/sft/domains/{domain}.jsonl"
        if data_path.exists():
            adapter = run_train(str(data_path), f"domain_{domain}")
            if adapter:
                trained_adapters.append(adapter)

    # 2. Train with different ranks on distilled patterns (24 adapters: 8 patterns x 3 ranks)
    print("\n" + "="*60)
    print("PHASE 2: Training distilled patterns with different ranks")
    print("="*60)

    patterns = ["mut_borrow_conflict", "double_mut_borrow", "return_local_ref", "option_ok_or",
                "result_map_err", "missing_clone", "missing_hash", "missing_ord"]
    ranks = [4, 16, 32]  # We already have rank=8, so train with 4, 16, 32

    for pattern in patterns:
        data_path = BASE_DIR / f"data/sft/distilled/{pattern}.jsonl"
        if not data_path.exists():
            continue
        for rank in ranks:
            adapter = run_train(str(data_path), f"distilled_{pattern}_r{rank}", rank=rank)
            if adapter:
                trained_adapters.append(adapter)

    # 3. Train on combined data with different configs (4 adapters)
    print("\n" + "="*60)
    print("PHASE 3: Training combined data with different configs")
    print("="*60)

    all_distilled = BASE_DIR / "data/sft/distilled/all_distilled.jsonl"
    if all_distilled.exists():
        configs = [
            ("combined_lr5e5", {"lr": 5e-5}),
            ("combined_lr2e4", {"lr": 2e-4}),
            ("combined_steps100", {"steps": 100}),
            ("combined_r16_steps50", {"rank": 16, "steps": 50}),
        ]
        for name, kwargs in configs:
            adapter = run_train(str(all_distilled), name, **kwargs)
            if adapter:
                trained_adapters.append(adapter)

    # 4. Train on existing pattern data with different ranks (20 adapters: sample 10 patterns x 2 ranks)
    print("\n" + "="*60)
    print("PHASE 4: Training pattern data with r=4 and r=16")
    print("="*60)

    pattern_files = list((BASE_DIR / "data/sft/patterns").glob("*.jsonl"))
    # Sample 10 patterns for variety
    sampled_patterns = random.sample(pattern_files, min(10, len(pattern_files)))

    for pattern_file in sampled_patterns:
        for rank in [4, 16]:
            adapter = run_train(str(pattern_file), f"pattern_{pattern_file.stem}_r{rank}", rank=rank)
            if adapter:
                trained_adapters.append(adapter)

    print("\n" + "="*60)
    print(f"COMPLETE: Trained {len(trained_adapters)} new adapters")
    print("="*60)

    # Save adapter list
    adapter_list = OUTPUT_BASE / "adapter_list.txt"
    with open(adapter_list, "w") as f:
        for adapter in trained_adapters:
            f.write(str(adapter) + "\n")
    print(f"Adapter list saved to: {adapter_list}")


if __name__ == "__main__":
    main()
