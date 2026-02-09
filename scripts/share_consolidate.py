#!/usr/bin/env python3
"""
Share-Style Adapter Consolidation

Based on arXiv:2602.06043v1 - "Shared LoRA Subspaces for almost Strict Continual Learning"

This script consolidates multiple LoRA adapters into a shared low-rank basis
with per-adapter coefficients. This enables:
1. Efficient storage (shared basis + small coefficients)
2. No forgetting (basis is frozen for new tasks)
3. Fast adaptation (only train coefficients for new tasks)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_lora_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors."""
    adapter_file = adapter_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_file}")
    return load_file(adapter_file)


def extract_delta_w(lora_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Extract delta_W = B @ A for each LoRA layer.

    LoRA stores: base_layer.lora_A.weight and base_layer.lora_B.weight
    delta_W = B @ A (the low-rank update)
    """
    delta_ws = {}

    # Group A and B weights by layer
    layers = {}
    for name, weight in lora_weights.items():
        if "lora_A" in name:
            layer_name = name.replace(".lora_A.weight", "")
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name]["A"] = weight
        elif "lora_B" in name:
            layer_name = name.replace(".lora_B.weight", "")
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name]["B"] = weight

    # Compute delta_W = B @ A for each layer
    for layer_name, weights in layers.items():
        if "A" in weights and "B" in weights:
            A = weights["A"]  # Shape: (r, in_features)
            B = weights["B"]  # Shape: (out_features, r)
            delta_W = B @ A   # Shape: (out_features, in_features)
            delta_ws[layer_name] = delta_W

    return delta_ws


def flatten_deltas(delta_ws: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[str], list[tuple]]:
    """
    Flatten all delta_W matrices into a single vector.

    Returns:
        - flattened: 1D numpy array of all weights
        - layer_names: list of layer names in order
        - shapes: list of original shapes for reconstruction
    """
    flattened = []
    layer_names = []
    shapes = []

    for name in sorted(delta_ws.keys()):
        tensor = delta_ws[name].float()  # Convert bfloat16 to float32
        flattened.append(tensor.cpu().numpy().flatten())
        layer_names.append(name)
        shapes.append(tensor.shape)

    return np.concatenate(flattened), layer_names, shapes


def unflatten_deltas(
    flattened: np.ndarray,
    layer_names: list[str],
    shapes: list[tuple]
) -> dict[str, torch.Tensor]:
    """Reconstruct delta_W dict from flattened vector."""
    delta_ws = {}
    offset = 0

    for name, shape in zip(layer_names, shapes):
        size = np.prod(shape)
        tensor = flattened[offset:offset + size].reshape(shape)
        delta_ws[name] = torch.from_numpy(tensor)
        offset += size

    return delta_ws


def consolidate_adapters(
    adapter_paths: list[Path],
    rank_k: int = 8
) -> tuple[np.ndarray, list[np.ndarray], list[str], list[tuple]]:
    """
    Consolidate multiple LoRA adapters into shared basis + coefficients.

    Algorithm (from Share paper):
    1. Extract delta_W from each adapter
    2. Flatten and stack into matrix
    3. Center the data
    4. SVD to find shared subspace
    5. Project each adapter onto subspace

    Args:
        adapter_paths: List of paths to adapter directories
        rank_k: Number of principal components to keep

    Returns:
        - basis: (k, d) matrix of basis vectors
        - coefficients: List of (k,) coefficient vectors, one per adapter
        - layer_names: Layer names for reconstruction
        - shapes: Original shapes for reconstruction
    """
    logger.info(f"Consolidating {len(adapter_paths)} adapters with rank {rank_k}")

    # Extract delta_W from each adapter
    all_deltas = []
    layer_names = None
    shapes = None

    for path in adapter_paths:
        logger.info(f"Loading adapter: {path}")
        weights = load_lora_weights(path)
        delta_ws = extract_delta_w(weights)
        flattened, names, shps = flatten_deltas(delta_ws)

        # Validate all adapters have same structure
        if layer_names is None:
            layer_names = names
            shapes = shps
        else:
            assert layer_names == names, "Adapters have different layer structures"

        all_deltas.append(flattened)

    # Stack into matrix: (n_adapters, d)
    stacked = np.stack(all_deltas)
    logger.info(f"Stacked shape: {stacked.shape}")

    # Center the data
    mean = stacked.mean(axis=0)
    centered = stacked - mean

    # SVD to find shared subspace
    logger.info("Computing SVD...")
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)

    # Keep top-k components
    basis = Vh[:rank_k]  # Shape: (k, d)
    logger.info(f"Basis shape: {basis.shape}")
    logger.info(f"Top {rank_k} singular values: {S[:rank_k]}")
    logger.info(f"Explained variance ratio: {(S[:rank_k]**2).sum() / (S**2).sum():.3f}")

    # Project each adapter onto basis
    coefficients = []
    for i, delta in enumerate(all_deltas):
        # Remove mean, project onto basis
        centered_delta = delta - mean
        coef = basis @ centered_delta  # Shape: (k,)
        coefficients.append(coef)
        logger.info(f"Adapter {i} coefficients: {coef[:4]}...")

    # Store mean as part of basis for reconstruction
    return basis, mean, coefficients, layer_names, shapes


def reconstruct_adapter(
    basis: np.ndarray,
    mean: np.ndarray,
    coefficients: np.ndarray,
    layer_names: list[str],
    shapes: list[tuple]
) -> dict[str, torch.Tensor]:
    """
    Reconstruct delta_W from basis and coefficients.

    delta_W = mean + basis.T @ coefficients
    """
    # Reconstruct flattened weights
    reconstructed = mean + basis.T @ coefficients

    # Unflatten to dict
    return unflatten_deltas(reconstructed, layer_names, shapes)


def reconstruction_error(
    original: dict[str, torch.Tensor],
    reconstructed: dict[str, torch.Tensor]
) -> float:
    """Compute mean squared error between original and reconstructed."""
    total_error = 0.0
    total_elements = 0

    for name in original:
        orig = original[name].float()
        recon = reconstructed[name].float()
        error = ((orig - recon) ** 2).sum().item()
        total_error += error
        total_elements += orig.numel()

    return total_error / total_elements


def save_consolidated(
    output_dir: Path,
    basis: np.ndarray,
    mean: np.ndarray,
    coefficients: list[np.ndarray],
    layer_names: list[str],
    shapes: list[tuple],
    adapter_names: list[str]
):
    """Save consolidated basis and coefficients."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save basis
    np.save(output_dir / "basis.npy", basis)
    np.save(output_dir / "mean.npy", mean)

    # Save coefficients (with cycle index to avoid name collisions)
    for i, (coef, name) in enumerate(zip(coefficients, adapter_names)):
        np.save(output_dir / f"coef_cycle{i}_{name}.npy", coef)

    # Save metadata
    metadata = {
        "layer_names": layer_names,
        "shapes": [list(s) for s in shapes],
        "adapter_names": adapter_names,
        "rank": basis.shape[0],
        "n_adapters": len(coefficients),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved consolidated adapters to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate LoRA adapters using Share method")
    parser.add_argument("--adapters", "-a", nargs="+", required=True,
                        help="Paths to adapter directories")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for consolidated adapters")
    parser.add_argument("--rank", "-r", type=int, default=8,
                        help="Rank of shared basis (default: 8)")
    args = parser.parse_args()

    adapter_paths = [Path(p) for p in args.adapters]
    output_dir = Path(args.output)

    # Validate adapters exist
    for path in adapter_paths:
        if not (path / "adapter_model.safetensors").exists():
            logger.error(f"Adapter not found: {path}")
            return 1

    # Consolidate
    basis, mean, coefficients, layer_names, shapes = consolidate_adapters(
        adapter_paths, rank_k=args.rank
    )

    # Compute reconstruction errors
    logger.info("\n=== Reconstruction Errors ===")
    for i, path in enumerate(adapter_paths):
        weights = load_lora_weights(path)
        original = extract_delta_w(weights)
        reconstructed = reconstruct_adapter(basis, mean, coefficients[i], layer_names, shapes)
        error = reconstruction_error(original, reconstructed)
        logger.info(f"Adapter {path.name}: MSE = {error:.6f}")

    # Save
    adapter_names = [p.name for p in adapter_paths]
    save_consolidated(output_dir, basis, mean, coefficients, layer_names, shapes, adapter_names)

    # Summary
    print("\n=== Consolidation Summary ===")
    print(f"Adapters consolidated: {len(adapter_paths)}")
    print(f"Shared basis rank: {args.rank}")
    print(f"Total parameters in basis: {basis.size:,}")
    print(f"Parameters per adapter (coefficients): {basis.shape[0]}")
    print(f"Compression ratio: {basis.size / (basis.shape[0] * len(adapter_paths)):.1f}x")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
