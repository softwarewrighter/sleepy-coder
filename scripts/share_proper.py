#!/usr/bin/env python3
"""
Proper Share Algorithm Implementation

Based on arXiv:2602.06043v1 - "Shared LoRA Subspaces for almost Strict Continual Learning"

KEY DIFFERENCE from previous implementation:
- Previous: Combined delta_W = B @ A, then SVD on stacked delta_Ws
- Correct: Keep B and A matrices SEPARATE, do SVD on each independently

Algorithm:
1. Phase 1 (Initialization): Extract k principal basis vectors from B and A separately
2. Phase 2 (Adaptation): Train only coefficient matrices (basis frozen)
3. Phase 3 (Merging): Reconstruct, stack, re-SVD to update basis
"""

import argparse
import json
import logging
from dataclasses import dataclass
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


@dataclass
class ShareBasis:
    """Shared basis vectors for B and A matrices."""
    beta: dict[str, np.ndarray]    # β^t: {layer_name: (n, k)} - basis for B matrices
    alpha: dict[str, np.ndarray]   # α^t: {layer_name: (d, k)} - basis for A matrices
    k: int                          # Number of principal components
    p: int                          # Pseudo-rank for coefficients


@dataclass
class TaskCoefficients:
    """Per-task coefficient matrices."""
    epsilon_beta: dict[str, np.ndarray]  # ε_β: {layer_name: (k, p)}
    epsilon_alpha: dict[str, np.ndarray] # ε_α: {layer_name: (k, p)}
    task_name: str


def load_lora_adapter(adapter_path: Path) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors."""
    adapter_file = adapter_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_file}")
    return load_file(adapter_file)


def extract_lora_matrices(weights: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """
    Extract B and A matrices for each layer.

    LoRA format:
        base_layer.lora_A.weight -> A matrix (r, in_features)
        base_layer.lora_B.weight -> B matrix (out_features, r)

    Returns:
        B_matrices: {layer_name: tensor(out_features, r)}
        A_matrices: {layer_name: tensor(r, in_features)}
    """
    B_matrices = {}
    A_matrices = {}

    # Group by layer
    layers = {}
    for name, weight in weights.items():
        if "lora_A" in name:
            layer_name = name.replace(".lora_A.weight", "")
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name]["A"] = weight.float().cpu().numpy()
        elif "lora_B" in name:
            layer_name = name.replace(".lora_B.weight", "")
            if layer_name not in layers:
                layers[layer_name] = {}
            layers[layer_name]["B"] = weight.float().cpu().numpy()

    for layer_name, mats in layers.items():
        if "A" in mats and "B" in mats:
            B_matrices[layer_name] = mats["B"]  # (out, r)
            A_matrices[layer_name] = mats["A"]  # (r, in)

    return B_matrices, A_matrices


def compute_explained_variance_k(S: np.ndarray, target_variance: float = 0.6) -> int:
    """
    Compute k such that top-k singular values explain target_variance of total variance.

    Paper: "60% explained variance" threshold for selecting k.
    """
    total_var = (S ** 2).sum()
    cumvar = np.cumsum(S ** 2) / total_var
    k = np.searchsorted(cumvar, target_variance) + 1
    return min(k, len(S))


def share_consolidate_phase1(
    adapter_paths: list[Path],
    k_target_variance: float = 0.6,
    min_k: int = 4,
    max_k: int = 32
) -> tuple[ShareBasis, list[TaskCoefficients], dict]:
    """
    Phase 1: Extract shared basis from multiple LoRA adapters.

    For each layer:
    1. Stack all B matrices horizontally: ℬ = [B_1, B_2, ..., B_T] ∈ ℝ^(n × T*r)
    2. Center and SVD: U, S, Vh = SVD(ℬ - mean)
    3. Take top-k left singular vectors: β = U[:, :k] ∈ ℝ^(n × k)
    4. Compute coefficients: ε_β^i = β^T @ B_i (projection)

    Same for A matrices.
    """
    logger.info(f"Phase 1: Consolidating {len(adapter_paths)} adapters")

    # Load all adapters
    all_Bs = []
    all_As = []
    adapter_names = []

    layer_names = None

    for path in adapter_paths:
        logger.info(f"Loading: {path}")
        weights = load_lora_adapter(path)
        B_mats, A_mats = extract_lora_matrices(weights)

        if layer_names is None:
            layer_names = sorted(B_mats.keys())
        else:
            assert sorted(B_mats.keys()) == layer_names, "Adapter layer mismatch"

        all_Bs.append(B_mats)
        all_As.append(A_mats)
        adapter_names.append(path.name)

    T = len(adapter_paths)  # Number of tasks/adapters

    # Process each layer
    beta = {}   # Basis for B matrices
    alpha = {}  # Basis for A matrices
    coefficients = []
    layer_info = {}

    for layer_name in layer_names:
        # Stack B matrices: ℬ = [B_1, ..., B_T] horizontally
        # Each B_i is (out, r), so ℬ is (out, T*r)
        B_stack = np.hstack([all_Bs[i][layer_name] for i in range(T)])

        # Stack A matrices: 𝒜 = [A_1, ..., A_T]
        # Each A_i is (r, in), we stack as (r*T, in) then transpose for SVD
        # Actually, paper stacks columns: 𝒜 ∈ ℝ^(d × T*r)
        # Since A is (r, d), we need A^T which is (d, r), then stack horizontally
        A_stack = np.hstack([all_As[i][layer_name].T for i in range(T)])  # (in, T*r)

        # SVD on B stack
        logger.debug(f"  {layer_name}: B_stack shape {B_stack.shape}")
        U_b, S_b, Vh_b = np.linalg.svd(B_stack, full_matrices=False)
        k_b = compute_explained_variance_k(S_b, k_target_variance)
        k_b = max(min_k, min(k_b, max_k, U_b.shape[1]))

        # β = top-k left singular vectors: (out, k)
        beta[layer_name] = U_b[:, :k_b]

        # SVD on A stack
        U_a, S_a, Vh_a = np.linalg.svd(A_stack, full_matrices=False)
        k_a = compute_explained_variance_k(S_a, k_target_variance)
        k_a = max(min_k, min(k_a, max_k, U_a.shape[1]))

        # α = top-k left singular vectors: (in, k)
        alpha[layer_name] = U_a[:, :k_a]

        layer_info[layer_name] = {
            "B_shape": B_stack.shape,
            "A_shape": A_stack.shape,
            "k_b": k_b,
            "k_a": k_a,
            "B_explained_var": float((S_b[:k_b]**2).sum() / (S_b**2).sum()),
            "A_explained_var": float((S_a[:k_a]**2).sum() / (S_a**2).sum()),
            "top_singular_B": S_b[:min(5, len(S_b))].tolist(),
            "top_singular_A": S_a[:min(5, len(S_a))].tolist(),
        }

    # Use consistent k across layers (take median)
    k_values = [layer_info[l]["k_b"] for l in layer_names]
    k = int(np.median(k_values))
    p = max(1, k // 3)  # Pseudo-rank (paper: p=1 or p=r/3)

    logger.info(f"Selected k={k}, p={p}")

    # Recompute with consistent k and compute coefficients
    for layer_name in layer_names:
        # Trim basis to consistent k
        beta[layer_name] = beta[layer_name][:, :k]
        alpha[layer_name] = alpha[layer_name][:, :k]

    # Compute coefficients for each adapter
    for i in range(T):
        epsilon_beta = {}
        epsilon_alpha = {}

        for layer_name in layer_names:
            B_i = all_Bs[i][layer_name]  # (out, r)
            A_i = all_As[i][layer_name]  # (r, in)

            # Project B onto basis: ε_β = β^T @ B_i
            # β is (out, k), B_i is (out, r) → ε_β is (k, r)
            eps_b = beta[layer_name].T @ B_i  # (k, r)

            # Project A onto basis: A_i^T is (in, r), α is (in, k)
            # ε_α = α^T @ A_i^T → (k, r)
            eps_a = alpha[layer_name].T @ A_i.T  # (k, r)

            # For pseudo-rank p < r, we'd need to compress further
            # For now, keep full r (p = r)
            epsilon_beta[layer_name] = eps_b
            epsilon_alpha[layer_name] = eps_a

        coefficients.append(TaskCoefficients(
            epsilon_beta=epsilon_beta,
            epsilon_alpha=epsilon_alpha,
            task_name=adapter_names[i]
        ))

    # Compute reconstruction errors
    logger.info("\n=== Reconstruction Errors ===")
    for i in range(T):
        total_error = 0.0
        total_elements = 0

        for layer_name in layer_names:
            B_orig = all_Bs[i][layer_name]
            A_orig = all_As[i][layer_name]

            # Reconstruct: B_hat = β @ ε_β
            B_hat = beta[layer_name] @ coefficients[i].epsilon_beta[layer_name]
            # Reconstruct: A_hat = (α @ ε_α)^T = ε_α^T @ α^T
            A_hat = (alpha[layer_name] @ coefficients[i].epsilon_alpha[layer_name]).T

            # Compute error on delta_W = B @ A
            delta_orig = B_orig @ A_orig
            delta_hat = B_hat @ A_hat

            error = ((delta_orig - delta_hat) ** 2).sum()
            total_error += error
            total_elements += delta_orig.size

        mse = total_error / total_elements
        logger.info(f"  Adapter {adapter_names[i]}: MSE = {mse:.6f}")

    basis = ShareBasis(beta=beta, alpha=alpha, k=k, p=p)

    return basis, coefficients, layer_info


def reconstruct_lora_adapter(
    basis: ShareBasis,
    coef: TaskCoefficients,
    layer_names: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Reconstruct B and A matrices from basis and coefficients.

    B_hat = β @ ε_β  →  (out, k) @ (k, r) = (out, r)
    A_hat = (α @ ε_α)^T  →  ((in, k) @ (k, r))^T = (r, in)
    """
    B_recon = {}
    A_recon = {}

    for layer_name in layer_names:
        B_hat = basis.beta[layer_name] @ coef.epsilon_beta[layer_name]
        A_hat = (basis.alpha[layer_name] @ coef.epsilon_alpha[layer_name]).T

        B_recon[layer_name] = B_hat
        A_recon[layer_name] = A_hat

    return B_recon, A_recon


def share_merge_phase3(
    basis: ShareBasis,
    existing_coefs: list[TaskCoefficients],
    new_B: dict[str, np.ndarray],
    new_A: dict[str, np.ndarray],
    new_task_name: str,
    k_target_variance: float = 0.6
) -> tuple[ShareBasis, list[TaskCoefficients]]:
    """
    Phase 3: Merge a new adapter into the shared subspace.

    1. Reconstruct all prior adapters from current basis + coefficients
    2. Stack with new adapter
    3. Re-run SVD to update basis
    4. Recalculate all coefficients analytically
    """
    logger.info(f"Phase 3: Merging new task '{new_task_name}'")

    layer_names = list(basis.beta.keys())
    T = len(existing_coefs)

    # Reconstruct all prior adapters
    all_Bs = []
    all_As = []

    for coef in existing_coefs:
        B_recon, A_recon = reconstruct_lora_adapter(basis, coef, layer_names)
        all_Bs.append(B_recon)
        all_As.append(A_recon)

    # Add new adapter
    all_Bs.append(new_B)
    all_As.append(new_A)

    task_names = [c.task_name for c in existing_coefs] + [new_task_name]

    # Re-run consolidation
    new_beta = {}
    new_alpha = {}
    new_coefs = []

    for layer_name in layer_names:
        # Stack B matrices
        B_stack = np.hstack([all_Bs[i][layer_name] for i in range(T + 1)])

        # Stack A matrices (transposed)
        A_stack = np.hstack([all_As[i][layer_name].T for i in range(T + 1)])

        # SVD for B
        U_b, S_b, Vh_b = np.linalg.svd(B_stack, full_matrices=False)
        k_b = compute_explained_variance_k(S_b, k_target_variance)
        k_b = min(k_b, basis.k + 2, U_b.shape[1])  # Allow slight growth

        new_beta[layer_name] = U_b[:, :k_b]

        # SVD for A
        U_a, S_a, Vh_a = np.linalg.svd(A_stack, full_matrices=False)
        k_a = min(compute_explained_variance_k(S_a, k_target_variance), k_b)

        new_alpha[layer_name] = U_a[:, :k_a]

    # Use consistent k
    k = min(new_beta[layer_names[0]].shape[1], new_alpha[layer_names[0]].shape[1])
    for layer_name in layer_names:
        new_beta[layer_name] = new_beta[layer_name][:, :k]
        new_alpha[layer_name] = new_alpha[layer_name][:, :k]

    # Recalculate coefficients analytically for all tasks
    for i in range(T + 1):
        epsilon_beta = {}
        epsilon_alpha = {}

        for layer_name in layer_names:
            B_i = all_Bs[i][layer_name]
            A_i = all_As[i][layer_name]

            # Analytical projection: ε_β = β^T @ B_i (when β is orthonormal)
            eps_b = new_beta[layer_name].T @ B_i
            eps_a = new_alpha[layer_name].T @ A_i.T

            epsilon_beta[layer_name] = eps_b
            epsilon_alpha[layer_name] = eps_a

        new_coefs.append(TaskCoefficients(
            epsilon_beta=epsilon_beta,
            epsilon_alpha=epsilon_alpha,
            task_name=task_names[i]
        ))

    new_basis = ShareBasis(beta=new_beta, alpha=new_alpha, k=k, p=basis.p)

    logger.info(f"Updated basis: k={k}")
    return new_basis, new_coefs


def save_share_model(
    output_dir: Path,
    basis: ShareBasis,
    coefficients: list[TaskCoefficients],
    layer_info: dict
):
    """Save Share model (basis + coefficients) to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save basis
    basis_dir = output_dir / "basis"
    basis_dir.mkdir(exist_ok=True)

    for layer_name in basis.beta.keys():
        safe_name = layer_name.replace(".", "_")
        np.save(basis_dir / f"beta_{safe_name}.npy", basis.beta[layer_name])
        np.save(basis_dir / f"alpha_{safe_name}.npy", basis.alpha[layer_name])

    # Save coefficients for each task
    coef_dir = output_dir / "coefficients"
    coef_dir.mkdir(exist_ok=True)

    for i, coef in enumerate(coefficients):
        task_dir = coef_dir / f"task_{i:03d}_{coef.task_name}"
        task_dir.mkdir(exist_ok=True)

        for layer_name in coef.epsilon_beta.keys():
            safe_name = layer_name.replace(".", "_")
            np.save(task_dir / f"eps_beta_{safe_name}.npy", coef.epsilon_beta[layer_name])
            np.save(task_dir / f"eps_alpha_{safe_name}.npy", coef.epsilon_alpha[layer_name])

    # Save metadata (convert numpy types to Python types)
    def convert_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        return obj

    metadata = {
        "k": int(basis.k),
        "p": int(basis.p),
        "num_tasks": len(coefficients),
        "task_names": [c.task_name for c in coefficients],
        "layer_names": list(basis.beta.keys()),
        "layer_info": convert_to_python(layer_info),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved Share model to {output_dir}")


def load_share_model(share_dir: Path) -> tuple[ShareBasis, list[TaskCoefficients], dict]:
    """Load Share model from disk."""
    with open(share_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load basis
    beta = {}
    alpha = {}
    basis_dir = share_dir / "basis"

    for layer_name in metadata["layer_names"]:
        safe_name = layer_name.replace(".", "_")
        beta[layer_name] = np.load(basis_dir / f"beta_{safe_name}.npy")
        alpha[layer_name] = np.load(basis_dir / f"alpha_{safe_name}.npy")

    basis = ShareBasis(
        beta=beta,
        alpha=alpha,
        k=metadata["k"],
        p=metadata["p"]
    )

    # Load coefficients
    coefficients = []
    coef_dir = share_dir / "coefficients"

    for i, task_name in enumerate(metadata["task_names"]):
        task_dir = coef_dir / f"task_{i:03d}_{task_name}"

        epsilon_beta = {}
        epsilon_alpha = {}

        for layer_name in metadata["layer_names"]:
            safe_name = layer_name.replace(".", "_")
            epsilon_beta[layer_name] = np.load(task_dir / f"eps_beta_{safe_name}.npy")
            epsilon_alpha[layer_name] = np.load(task_dir / f"eps_alpha_{safe_name}.npy")

        coefficients.append(TaskCoefficients(
            epsilon_beta=epsilon_beta,
            epsilon_alpha=epsilon_alpha,
            task_name=task_name
        ))

    return basis, coefficients, metadata


def export_to_lora_adapter(
    basis: ShareBasis,
    coef: TaskCoefficients,
    output_path: Path,
    original_adapter_path: Optional[Path] = None
):
    """
    Export reconstructed adapter as standard LoRA format.

    This allows using the Share model with standard LoRA tools.
    """
    layer_names = list(basis.beta.keys())
    B_recon, A_recon = reconstruct_lora_adapter(basis, coef, layer_names)

    # Convert to safetensors format (must be contiguous)
    tensors = {}
    for layer_name in layer_names:
        tensors[f"{layer_name}.lora_B.weight"] = torch.from_numpy(B_recon[layer_name]).to(torch.bfloat16).contiguous()
        tensors[f"{layer_name}.lora_A.weight"] = torch.from_numpy(A_recon[layer_name]).to(torch.bfloat16).contiguous()

    output_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path / "adapter_model.safetensors")

    # Copy adapter_config.json if available
    if original_adapter_path and (original_adapter_path / "adapter_config.json").exists():
        import shutil
        shutil.copy(
            original_adapter_path / "adapter_config.json",
            output_path / "adapter_config.json"
        )

    logger.info(f"Exported LoRA adapter to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Share: Proper LoRA subspace consolidation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Consolidate multiple adapters
  python share_proper.py consolidate -a adapter1 adapter2 adapter3 -o share_model

  # Merge a new adapter into existing Share model
  python share_proper.py merge --share share_model --new-adapter adapter4

  # Export a task's adapter from Share model
  python share_proper.py export --share share_model --task 0 -o exported_adapter
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Consolidate command
    cons_parser = subparsers.add_parser("consolidate", help="Consolidate adapters into Share model")
    cons_parser.add_argument("--adapters", "-a", nargs="+", required=True, help="Adapter directories")
    cons_parser.add_argument("--output", "-o", required=True, help="Output directory")
    cons_parser.add_argument("--variance", "-v", type=float, default=0.6, help="Target explained variance (default: 0.6)")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge new adapter into Share model")
    merge_parser.add_argument("--share", "-s", required=True, help="Existing Share model directory")
    merge_parser.add_argument("--new-adapter", "-n", required=True, help="New adapter to merge")
    merge_parser.add_argument("--output", "-o", help="Output directory (default: update in place)")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export task adapter from Share model")
    export_parser.add_argument("--share", "-s", required=True, help="Share model directory")
    export_parser.add_argument("--task", "-t", type=int, default=0, help="Task index to export")
    export_parser.add_argument("--output", "-o", required=True, help="Output adapter directory")
    export_parser.add_argument("--original", help="Original adapter for config file")

    args = parser.parse_args()

    if args.command == "consolidate":
        adapter_paths = [Path(p) for p in args.adapters]
        output_dir = Path(args.output)

        basis, coefficients, layer_info = share_consolidate_phase1(
            adapter_paths,
            k_target_variance=args.variance
        )

        save_share_model(output_dir, basis, coefficients, layer_info)

        # Summary
        print("\n=== Share Consolidation Summary ===")
        print(f"Adapters consolidated: {len(adapter_paths)}")
        print(f"Principal components k: {basis.k}")
        print(f"Pseudo-rank p: {basis.p}")
        print(f"Layers: {len(basis.beta)}")

        # Compute storage savings
        total_basis_params = sum(
            basis.beta[l].size + basis.alpha[l].size
            for l in basis.beta
        )
        total_coef_params = sum(
            c.epsilon_beta[l].size + c.epsilon_alpha[l].size
            for c in coefficients
            for l in c.epsilon_beta
        )
        print(f"\nStorage:")
        print(f"  Basis parameters: {total_basis_params:,}")
        print(f"  Per-task coefficients: {total_coef_params // len(coefficients):,}")
        print(f"  Total: {total_basis_params + total_coef_params:,}")
        print(f"\nOutput: {output_dir}")

    elif args.command == "merge":
        share_dir = Path(args.share)
        new_adapter = Path(args.new_adapter)
        output_dir = Path(args.output) if args.output else share_dir

        # Load existing Share model
        basis, coefficients, metadata = load_share_model(share_dir)

        # Load new adapter
        weights = load_lora_adapter(new_adapter)
        new_B, new_A = extract_lora_matrices(weights)

        # Merge
        new_basis, new_coefs = share_merge_phase3(
            basis, coefficients, new_B, new_A, new_adapter.name
        )

        # Save
        save_share_model(output_dir, new_basis, new_coefs, metadata.get("layer_info", {}))

        print(f"\nMerged {new_adapter.name} into Share model")
        print(f"Total tasks: {len(new_coefs)}")
        print(f"Output: {output_dir}")

    elif args.command == "export":
        share_dir = Path(args.share)
        output_path = Path(args.output)

        basis, coefficients, metadata = load_share_model(share_dir)

        if args.task >= len(coefficients):
            print(f"Error: Task index {args.task} out of range (0-{len(coefficients)-1})")
            return 1

        coef = coefficients[args.task]
        original = Path(args.original) if args.original else None

        export_to_lora_adapter(basis, coef, output_path, original)

        print(f"\nExported task '{coef.task_name}' to {output_path}")


if __name__ == "__main__":
    main()
