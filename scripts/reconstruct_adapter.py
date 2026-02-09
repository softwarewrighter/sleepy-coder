#!/usr/bin/env python3
"""
Reconstruct LoRA adapter from Share-style basis + coefficients.

This is the key innovation: create new adapters by combining
coefficients instead of full training.

Usage:
    python reconstruct_adapter.py --basis runs/shared_basis_v2 \
                                  --coef runs/shared_basis_v2/coef_strategy1.npy \
                                  --output runs/adapters/reconstructed
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_basis(basis_dir: Path):
    """Load shared basis, mean, and metadata."""
    basis = np.load(basis_dir / "basis.npy")
    mean = np.load(basis_dir / "mean.npy")

    with open(basis_dir / "metadata.json") as f:
        metadata = json.load(f)

    return basis, mean, metadata


def reconstruct_delta_w(basis: np.ndarray, mean: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Reconstruct flattened delta_W from basis and coefficients."""
    # delta_W = mean + basis.T @ coef
    return mean + basis.T @ coef


def unflatten_to_lora(
    delta_w: np.ndarray,
    layer_names: list,
    shapes: list,
    rank: int = 8
) -> dict:
    """
    Convert delta_W back to LoRA A/B matrices.

    For each layer's delta_W (shape: out x in), decompose as B @ A
    where A is (rank x in) and B is (out x rank).

    Uses SVD to find best rank-r approximation.
    """
    lora_weights = {}
    offset = 0

    for layer_name, shape in zip(layer_names, shapes):
        size = int(np.prod(shape))
        layer_delta = delta_w[offset:offset + size].reshape(shape)
        offset += size

        # SVD decomposition: delta_W ≈ U @ S @ Vh
        # LoRA: B @ A where B = U[:, :r] @ diag(sqrt(S[:r])), A = diag(sqrt(S[:r])) @ Vh[:r, :]
        U, S, Vh = np.linalg.svd(layer_delta, full_matrices=False)

        # Take top-r components
        r = min(rank, len(S))
        sqrt_S = np.sqrt(S[:r])

        # B is (out_features, r), A is (r, in_features)
        B = U[:, :r] * sqrt_S  # Broadcast sqrt_S across columns
        A = (Vh[:r, :].T * sqrt_S).T  # Broadcast sqrt_S across rows

        # Convert to torch tensors
        lora_weights[f"{layer_name}.lora_A.weight"] = torch.from_numpy(A.astype(np.float32))
        lora_weights[f"{layer_name}.lora_B.weight"] = torch.from_numpy(B.astype(np.float32))

    return lora_weights


def create_adapter_config(rank: int, layer_names: list) -> dict:
    """Create PEFT adapter config."""
    # Extract target modules from layer names
    target_modules = set()
    for name in layer_names:
        # e.g., "model.layers.0.self_attn.q_proj" -> "q_proj"
        module = name.split(".")[-1]
        target_modules.add(module)

    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False
    }


def main():
    parser = argparse.ArgumentParser(description="Reconstruct adapter from basis + coefficients")
    parser.add_argument("--basis", "-b", required=True, help="Path to shared basis directory")
    parser.add_argument("--coef", "-c", required=True, help="Path to coefficient file (.npy)")
    parser.add_argument("--output", "-o", required=True, help="Output adapter directory")
    parser.add_argument("--rank", "-r", type=int, default=8, help="LoRA rank for reconstruction")
    args = parser.parse_args()

    basis_dir = Path(args.basis)
    coef_path = Path(args.coef)
    output_dir = Path(args.output)

    # Load basis and metadata
    logger.info(f"Loading basis from {basis_dir}")
    basis, mean, metadata = load_basis(basis_dir)
    logger.info(f"Basis shape: {basis.shape}")

    # Load coefficients
    logger.info(f"Loading coefficients from {coef_path}")
    coef = np.load(coef_path)
    logger.info(f"Coefficients: {coef[:4]}...")

    # Reconstruct delta_W
    logger.info("Reconstructing delta_W")
    delta_w = reconstruct_delta_w(basis, mean, coef)
    logger.info(f"delta_W shape: {delta_w.shape}, mean: {delta_w.mean():.6f}, std: {delta_w.std():.6f}")

    # Convert to LoRA weights
    logger.info(f"Converting to LoRA format (rank={args.rank})")
    layer_names = metadata["layer_names"]
    shapes = [tuple(s) for s in metadata["shapes"]]
    lora_weights = unflatten_to_lora(delta_w, layer_names, shapes, rank=args.rank)
    logger.info(f"Created {len(lora_weights)} LoRA weight tensors")

    # Save adapter
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    save_file(lora_weights, output_dir / "adapter_model.safetensors")
    logger.info(f"Saved weights to {output_dir / 'adapter_model.safetensors'}")

    # Save config
    config = create_adapter_config(args.rank, layer_names)
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {output_dir / 'adapter_config.json'}")

    # Save coefficient source for provenance
    provenance = {
        "basis_dir": str(basis_dir),
        "coefficient_file": str(coef_path),
        "coefficients": coef.tolist(),
        "reconstruction_rank": args.rank,
    }
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    logger.info(f"\nAdapter reconstructed successfully!")
    logger.info(f"Output: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Merge: python scripts/merge.py --adapter {output_dir}")
    logger.info(f"  2. Quantize and create Ollama model")
    logger.info(f"  3. Evaluate with gate check")


if __name__ == "__main__":
    main()
