#!/usr/bin/env python3
"""
Export Share model with averaged coefficients across all tasks.

The Share paper keeps separate coefficients per task for continual learning.
For general-purpose use, we need to average coefficients to get a single
adapter that works across all patterns.
"""
import os
import json
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from safetensors.torch import save_file
import shutil

PROJECT_ROOT = Path(__file__).parent.parent


def load_share_model(share_dir: Path):
    """Load Share model (basis + all coefficients)."""
    with open(share_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load basis
    basis_dir = share_dir / "basis"
    beta = {}
    alpha = {}

    for layer_name in metadata["layer_names"]:
        safe_name = layer_name.replace(".", "_")
        beta[layer_name] = np.load(basis_dir / f"beta_{safe_name}.npy")
        alpha[layer_name] = np.load(basis_dir / f"alpha_{safe_name}.npy")

    # Load all coefficients
    coef_dir = share_dir / "coefficients"
    all_eps_beta = []
    all_eps_alpha = []

    for i, task_name in enumerate(metadata["task_names"]):
        task_dir = coef_dir / f"task_{i:03d}_{task_name}"

        eps_beta = {}
        eps_alpha = {}

        for layer_name in metadata["layer_names"]:
            safe_name = layer_name.replace(".", "_")
            eps_beta[layer_name] = np.load(task_dir / f"eps_beta_{safe_name}.npy")
            eps_alpha[layer_name] = np.load(task_dir / f"eps_alpha_{safe_name}.npy")

        all_eps_beta.append(eps_beta)
        all_eps_alpha.append(eps_alpha)

    return beta, alpha, all_eps_beta, all_eps_alpha, metadata


def average_coefficients(all_eps_beta, all_eps_alpha, layer_names):
    """Average coefficients across all tasks."""
    avg_eps_beta = {}
    avg_eps_alpha = {}

    for layer_name in layer_names:
        # Stack and average
        beta_stack = np.stack([eps[layer_name] for eps in all_eps_beta])
        alpha_stack = np.stack([eps[layer_name] for eps in all_eps_alpha])

        avg_eps_beta[layer_name] = beta_stack.mean(axis=0)
        avg_eps_alpha[layer_name] = alpha_stack.mean(axis=0)

    return avg_eps_beta, avg_eps_alpha


def reconstruct_and_export(beta, alpha, eps_beta, eps_alpha, output_path, original_adapter):
    """Reconstruct LoRA adapter and export."""
    tensors = {}

    for layer_name in beta.keys():
        # B_hat = β @ ε_β  →  (out, k) @ (k, r) = (out, r)
        B_hat = beta[layer_name] @ eps_beta[layer_name]

        # A_hat = (α @ ε_α)^T  →  ((in, k) @ (k, r))^T = (r, in)
        A_hat = (alpha[layer_name] @ eps_alpha[layer_name]).T

        tensors[f"{layer_name}.lora_B.weight"] = torch.from_numpy(B_hat).to(torch.bfloat16).contiguous()
        tensors[f"{layer_name}.lora_A.weight"] = torch.from_numpy(A_hat).to(torch.bfloat16).contiguous()

    output_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path / "adapter_model.safetensors")

    # Copy adapter_config.json
    if original_adapter.exists():
        config_src = original_adapter / "adapter_config.json"
        if config_src.exists():
            shutil.copy(config_src, output_path / "adapter_config.json")

    print(f"Exported to: {output_path}")


def main():
    share_dir = PROJECT_ROOT / "runs" / "share" / "pattern_share_20260210_144114"
    output_path = PROJECT_ROOT / "runs" / "adapters" / "share_averaged"

    # Find an original adapter for config
    original_adapter = PROJECT_ROOT / "runs" / "adapters" / "pattern_adapters" / "missing_clone" / "20260210_130943" / "20260210_130949" / "adapter"

    print("Loading Share model...")
    beta, alpha, all_eps_beta, all_eps_alpha, metadata = load_share_model(share_dir)

    print(f"Tasks: {len(metadata['task_names'])}")
    print(f"Layers: {len(metadata['layer_names'])}")
    print(f"k: {metadata['k']}, p: {metadata['p']}")

    print("\nAveraging coefficients across all tasks...")
    avg_eps_beta, avg_eps_alpha = average_coefficients(
        all_eps_beta, all_eps_alpha, metadata["layer_names"]
    )

    print("Reconstructing and exporting averaged adapter...")
    reconstruct_and_export(beta, alpha, avg_eps_beta, avg_eps_alpha, output_path, original_adapter)

    print("\nDone! Next steps:")
    print(f"  1. Merge: python cuda/scripts/merge.py --adapter {output_path} --model-name sleepy-coder-share-avg")
    print("  2. Eval:  cd rust && cargo run --release -- eval --cycle 15 --model sleepy-coder-share-avg")


if __name__ == "__main__":
    main()
