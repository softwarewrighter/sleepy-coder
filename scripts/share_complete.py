#!/usr/bin/env python3
"""
Complete Implementation of Share Algorithm + UWSH Concepts

Papers:
- Share: arXiv:2602.06043 - "Shared LoRA Subspaces for almost Strict Continual Learning"
- UWSH: arXiv:2512.05117 - "Universal Weight Subspace Hypothesis"

Key Concepts from UWSH:
- Neural networks converge to shared spectral subspaces
- Few principal directions capture majority variance
- This is architecture-dependent, not task-dependent

Key Concepts from Share:
- Phase 1: Extract shared basis from existing adapters via SVD
- Phase 2: Train ONLY coefficients (tiny!) with frozen basis
- Phase 3: Merge new knowledge and update basis

Critical Implementation Details:
- Coefficients are (k, p) where p ≈ 1, NOT (k, r)
- Basis is FROZEN during Phase 2 training
- ~460x fewer parameters per new task vs full LoRA
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"

# ==============================================================================
# Paper Hyperparameters
# ==============================================================================

@dataclass
class ShareConfig:
    """Hyperparameters from Share paper (arXiv:2602.06043)"""
    variance_threshold: float = 0.6  # k selection: 60% explained variance
    pseudo_rank: int = 1  # p = 1 is effective (paper says p=r/3 for higher)
    phi_expansion: int = 2  # φ temporary factors for Phase 2 (paper: [1, k/4])
    learning_rate: float = 1e-4
    training_steps: int = 100
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


# ==============================================================================
# Share Model: Stores Basis + Per-Task Coefficients
# ==============================================================================

class ShareModel:
    """
    Complete Share model implementing both papers.

    Structure per layer:
    - beta: (n, k_beta) - shared basis for B matrices
    - alpha: (d, k_alpha) - shared basis for A matrices
    - coefficients[task]: (eps_beta, eps_alpha) where:
        - eps_beta: (k_beta, p) - per-task coefficients for B
        - eps_alpha: (k_alpha, p) - per-task coefficients for A

    Reconstruction:
        B_hat = beta @ eps_beta  -> (n, k) @ (k, p) = (n, p)
        A_hat = (alpha @ eps_alpha).T  -> (p, d)
        delta_W = B_hat @ A_hat  -> (n, p) @ (p, d) = (n, d)
    """

    def __init__(self, config: ShareConfig = None):
        self.config = config or ShareConfig()
        self.layer_names: List[str] = []
        self.r: int = 8  # Original LoRA rank

        # Shared basis (FROZEN during Phase 2)
        self.beta: Dict[str, np.ndarray] = {}  # layer -> (n, k_beta)
        self.alpha: Dict[str, np.ndarray] = {}  # layer -> (d, k_alpha)

        # Per-task coefficients (TRAINED during Phase 2)
        self.coefficients: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        # task_id -> {layer -> (eps_beta, eps_alpha)}
        # eps_beta: (k_beta, p), eps_alpha: (k_alpha, p)

        # Metadata
        self.task_ids: List[str] = []
        self.k_beta: Dict[str, int] = {}  # k per layer for beta
        self.k_alpha: Dict[str, int] = {}  # k per layer for alpha

    def save(self, output_dir: Path):
        """Save Share model to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "layer_names": self.layer_names,
            "r": self.r,
            "p": self.config.pseudo_rank,
            "variance_threshold": self.config.variance_threshold,
            "task_ids": self.task_ids,
            "k_beta": self.k_beta,
            "k_alpha": self.k_alpha,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save basis
        basis_dir = output_dir / "basis"
        basis_dir.mkdir(exist_ok=True)
        for layer in self.layer_names:
            safe = layer.replace(".", "_")
            np.save(basis_dir / f"beta_{safe}.npy", self.beta[layer])
            np.save(basis_dir / f"alpha_{safe}.npy", self.alpha[layer])

        # Save coefficients
        coef_dir = output_dir / "coefficients"
        for task_id, task_coefs in self.coefficients.items():
            task_dir = coef_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            for layer, (eps_beta, eps_alpha) in task_coefs.items():
                safe = layer.replace(".", "_")
                np.save(task_dir / f"eps_beta_{safe}.npy", eps_beta)
                np.save(task_dir / f"eps_alpha_{safe}.npy", eps_alpha)

        logger.info(f"Saved Share model to {output_dir}")
        logger.info(f"  Tasks: {len(self.task_ids)}")
        logger.info(f"  Layers: {len(self.layer_names)}")
        sample_layer = self.layer_names[0]
        logger.info(f"  Sample k_beta: {self.k_beta[sample_layer]}, k_alpha: {self.k_alpha[sample_layer]}")

    def load(self, share_dir: Path):
        """Load Share model from disk."""
        share_dir = Path(share_dir)

        with open(share_dir / "metadata.json") as f:
            metadata = json.load(f)

        self.layer_names = metadata["layer_names"]
        self.r = metadata.get("r", 8)
        self.config.pseudo_rank = metadata.get("p", 1)
        self.config.variance_threshold = metadata.get("variance_threshold", 0.6)
        self.task_ids = metadata.get("task_ids", [])
        self.k_beta = metadata.get("k_beta", {})
        self.k_alpha = metadata.get("k_alpha", {})

        # Load basis
        basis_dir = share_dir / "basis"
        for layer in self.layer_names:
            safe = layer.replace(".", "_")
            self.beta[layer] = np.load(basis_dir / f"beta_{safe}.npy")
            self.alpha[layer] = np.load(basis_dir / f"alpha_{safe}.npy")
            if layer not in self.k_beta:
                self.k_beta[layer] = self.beta[layer].shape[1]
                self.k_alpha[layer] = self.alpha[layer].shape[1]

        # Load coefficients
        coef_dir = share_dir / "coefficients"
        if coef_dir.exists():
            for task_dir in coef_dir.iterdir():
                if task_dir.is_dir():
                    task_id = task_dir.name
                    self.coefficients[task_id] = {}
                    for layer in self.layer_names:
                        safe = layer.replace(".", "_")
                        eps_beta = np.load(task_dir / f"eps_beta_{safe}.npy")
                        eps_alpha = np.load(task_dir / f"eps_alpha_{safe}.npy")
                        self.coefficients[task_id][layer] = (eps_beta, eps_alpha)
                    if task_id not in self.task_ids:
                        self.task_ids.append(task_id)

        logger.info(f"Loaded Share model: {len(self.layer_names)} layers, {len(self.task_ids)} tasks")
        sample = self.layer_names[0]
        logger.info(f"  {sample}: beta={self.beta[sample].shape}, alpha={self.alpha[sample].shape}")

    def reconstruct_lora(self, task_id: str) -> Dict[str, torch.Tensor]:
        """
        Reconstruct LoRA adapter tensors for a specific task.

        B_hat = beta @ eps_beta  -> (n, p)
        A_hat = (alpha @ eps_alpha).T  -> (p, d)
        """
        tensors = {}
        task_coefs = self.coefficients[task_id]

        for layer in self.layer_names:
            eps_beta, eps_alpha = task_coefs[layer]
            beta = self.beta[layer]  # (n, k_beta)
            alpha = self.alpha[layer]  # (d, k_alpha)

            B_hat = beta @ eps_beta  # (n, k) @ (k, p) = (n, p)
            A_hat = (alpha @ eps_alpha).T  # ((d, k) @ (k, p)).T = (p, d)

            tensors[f"{layer}.lora_B.weight"] = torch.from_numpy(B_hat).to(torch.bfloat16)
            tensors[f"{layer}.lora_A.weight"] = torch.from_numpy(A_hat).to(torch.bfloat16)

        return tensors

    def get_total_params(self) -> int:
        """Count total parameters in basis + all coefficients."""
        basis_params = sum(
            self.beta[l].size + self.alpha[l].size
            for l in self.layer_names
        )
        coef_params = sum(
            sum(eb.size + ea.size for eb, ea in task.values())
            for task in self.coefficients.values()
        )
        return basis_params + coef_params

    def get_params_per_task(self) -> int:
        """Count parameters needed for one new task (coefficient-only)."""
        if not self.task_ids:
            return 0
        task = self.coefficients[self.task_ids[0]]
        return sum(eb.size + ea.size for eb, ea in task.values())


# ==============================================================================
# Phase 1: Initialize Shared Basis from Existing Adapters
# ==============================================================================

def phase1_initialize(
    adapter_dirs: List[Path],
    config: ShareConfig = None,
) -> ShareModel:
    """
    Phase 1: Extract shared basis from existing LoRA adapters.

    From Share paper:
    1. Stack all B matrices: B = [B_1, B_2, ..., B_T] -> (n, T*r)
    2. Stack all A matrices: A = [A_1, A_2, ..., A_T] -> (d, T*r)
    3. Mean-center and SVD
    4. Select k by 60% explained variance threshold
    5. Compute per-adapter coefficients

    UWSH insight: The resulting basis captures the "universal subspace"
    that all these tasks naturally occupy.
    """
    config = config or ShareConfig()
    model = ShareModel(config)

    logger.info("=" * 60)
    logger.info("Phase 1: Initialize Shared Basis")
    logger.info("=" * 60)
    logger.info(f"Input adapters: {len(adapter_dirs)}")
    logger.info(f"Variance threshold: {config.variance_threshold}")
    logger.info(f"Pseudo-rank p: {config.pseudo_rank}")

    # Load all adapters
    adapters = []
    for adapter_dir in adapter_dirs:
        adapter_dir = Path(adapter_dir)
        safetensor_file = adapter_dir / "adapter_model.safetensors"
        if safetensor_file.exists():
            adapters.append(load_file(safetensor_file))
        else:
            logger.warning(f"No safetensors found in {adapter_dir}")

    if not adapters:
        raise ValueError("No adapters loaded")

    logger.info(f"Loaded {len(adapters)} adapters")

    # Get layer names from first adapter
    layer_names = set()
    for key in adapters[0].keys():
        if ".lora_B.weight" in key:
            layer_name = key.replace(".lora_B.weight", "")
            layer_names.add(layer_name)

    model.layer_names = sorted(layer_names)
    model.r = adapters[0][f"{model.layer_names[0]}.lora_B.weight"].shape[1]

    logger.info(f"Layers: {len(model.layer_names)}, r={model.r}")

    p = config.pseudo_rank

    # Process each layer
    for layer in model.layer_names:
        # Stack B matrices: (n, T*r)
        B_stack = torch.cat([
            adapter[f"{layer}.lora_B.weight"]  # (n, r)
            for adapter in adapters
        ], dim=1).float().numpy()

        # Stack A matrices transposed: (d, T*r)
        # A is stored as (r, d), we want columns, so transpose each
        A_stack = torch.cat([
            adapter[f"{layer}.lora_A.weight"].T  # (d, r)
            for adapter in adapters
        ], dim=1).float().numpy()

        n, _ = B_stack.shape
        d, _ = A_stack.shape

        # Mean-center
        B_mean = B_stack.mean(axis=1, keepdims=True)
        A_mean = A_stack.mean(axis=1, keepdims=True)
        B_centered = B_stack - B_mean
        A_centered = A_stack - A_mean

        # SVD for B basis
        U_b, S_b, _ = np.linalg.svd(B_centered, full_matrices=False)
        var_b = (S_b ** 2).cumsum() / (S_b ** 2).sum()
        k_beta = int(np.searchsorted(var_b, config.variance_threshold) + 1)
        k_beta = max(p, min(k_beta, len(S_b)))  # At least p, at most full rank

        # SVD for A basis
        U_a, S_a, _ = np.linalg.svd(A_centered, full_matrices=False)
        var_a = (S_a ** 2).cumsum() / (S_a ** 2).sum()
        k_alpha = int(np.searchsorted(var_a, config.variance_threshold) + 1)
        k_alpha = max(p, min(k_alpha, len(S_a)))

        # Store basis (left singular vectors)
        model.beta[layer] = U_b[:, :k_beta].astype(np.float32)  # (n, k_beta)
        model.alpha[layer] = U_a[:, :k_alpha].astype(np.float32)  # (d, k_alpha)
        model.k_beta[layer] = k_beta
        model.k_alpha[layer] = k_alpha

    # Compute coefficients for each adapter
    for i, adapter in enumerate(adapters):
        task_id = f"init_adapter_{i:03d}"
        model.task_ids.append(task_id)
        model.coefficients[task_id] = {}

        for layer in model.layer_names:
            B = adapter[f"{layer}.lora_B.weight"].float().numpy()  # (n, r)
            A = adapter[f"{layer}.lora_A.weight"].float().numpy()  # (r, d)

            beta = model.beta[layer]  # (n, k_beta)
            alpha = model.alpha[layer]  # (d, k_alpha)

            # Project B onto beta basis: eps_beta = beta.T @ B -> (k_beta, r)
            # Then reduce to pseudo-rank p
            eps_beta_full = beta.T @ B  # (k_beta, r)
            eps_beta = eps_beta_full[:, :p]  # (k_beta, p)

            # Project A.T onto alpha basis: eps_alpha = alpha.T @ A.T -> (k_alpha, r)
            eps_alpha_full = alpha.T @ A.T  # (k_alpha, r)
            eps_alpha = eps_alpha_full[:, :p]  # (k_alpha, p)

            model.coefficients[task_id][layer] = (
                eps_beta.astype(np.float32),
                eps_alpha.astype(np.float32)
            )

    # Log statistics
    total_k_beta = sum(model.k_beta.values())
    total_k_alpha = sum(model.k_alpha.values())
    params_per_task = model.get_params_per_task()
    full_lora_params = len(model.layer_names) * (model.beta[model.layer_names[0]].shape[0] * model.r +
                                                   model.alpha[model.layer_names[0]].shape[0] * model.r)

    logger.info(f"Phase 1 complete:")
    logger.info(f"  Total k_beta: {total_k_beta}, k_alpha: {total_k_alpha}")
    logger.info(f"  Params per new task: {params_per_task:,} (coefficient-only)")
    logger.info(f"  Full LoRA params: {full_lora_params:,}")
    logger.info(f"  Reduction: {full_lora_params / params_per_task:.1f}x fewer params per task")

    return model


# ==============================================================================
# Phase 2: Coefficient-Only Training (THE KEY TO PREVENTING FORGETTING)
# ==============================================================================

class CoefficientTrainer:
    """
    Train ONLY coefficients with frozen basis.

    This is the KEY innovation from Share:
    - Basis (beta, alpha) is FROZEN
    - Only tiny coefficient matrices (eps_beta, eps_alpha) are trained
    - This means ~460x fewer parameters than full LoRA
    - Much less capacity to overwrite existing knowledge
    """

    def __init__(
        self,
        share_model: ShareModel,
        base_model_name: str,
        device: str = "cuda",
    ):
        self.share = share_model
        self.device = device
        self.config = share_model.config

        # Load base model and tokenizer
        logger.info(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.base_model.eval()

        # Convert basis to tensors (FROZEN)
        self.beta_tensors = {
            layer: torch.from_numpy(self.share.beta[layer]).to(device).to(torch.bfloat16)
            for layer in self.share.layer_names
        }
        self.alpha_tensors = {
            layer: torch.from_numpy(self.share.alpha[layer]).to(device).to(torch.bfloat16)
            for layer in self.share.layer_names
        }

        # Trainable coefficients (initialized fresh for each new task)
        self.eps_beta: Dict[str, nn.Parameter] = {}
        self.eps_alpha: Dict[str, nn.Parameter] = {}

        # Get layer dimensions
        self.layer_dims = {}
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'weight') and any(layer in name for layer in self.share.layer_names):
                for layer in self.share.layer_names:
                    if layer in name and layer not in self.layer_dims:
                        self.layer_dims[layer] = module.weight.shape

        # Hook handles
        self.hooks = []

    def initialize_coefficients(self, phi: int = None):
        """
        Initialize fresh coefficients for a new task.

        phi: temporary expansion factor (paper suggests [1, k/4])
        """
        phi = phi or self.config.phi_expansion
        p = self.config.pseudo_rank

        self.eps_beta.clear()
        self.eps_alpha.clear()

        total_params = 0
        for layer in self.share.layer_names:
            k_beta = self.share.k_beta[layer]
            k_alpha = self.share.k_alpha[layer]

            # Initialize with small random values
            # Paper: ε ~ N(0, σ²) with small σ
            self.eps_beta[layer] = nn.Parameter(
                torch.randn(k_beta, p, device=self.device, dtype=torch.float32) * 0.01
            )
            self.eps_alpha[layer] = nn.Parameter(
                torch.randn(k_alpha, p, device=self.device, dtype=torch.float32) * 0.01
            )

            total_params += k_beta * p + k_alpha * p

        logger.info(f"Initialized {total_params:,} trainable coefficient parameters")
        return total_params

    def _get_delta_w(self, layer: str) -> torch.Tensor:
        """
        Compute delta_W = B_hat @ A_hat for a layer.

        B_hat = beta @ eps_beta  -> (n, p)
        A_hat = (alpha @ eps_alpha).T  -> (p, d)
        delta_W = B_hat @ A_hat  -> (n, d)
        """
        beta = self.beta_tensors[layer]  # (n, k_beta)
        alpha = self.alpha_tensors[layer]  # (d, k_alpha)
        eps_beta = self.eps_beta[layer].to(torch.bfloat16)  # (k_beta, p)
        eps_alpha = self.eps_alpha[layer].to(torch.bfloat16)  # (k_alpha, p)

        B_hat = beta @ eps_beta  # (n, p)
        A_hat = (alpha @ eps_alpha).T  # (p, d)

        return B_hat @ A_hat  # (n, d)

    def _create_forward_hook(self, layer: str):
        """Create a forward hook that adds delta_W to the layer output."""
        def hook(module, input, output):
            if not self.training:
                return output

            delta_W = self._get_delta_w(layer)
            # output shape: (batch, seq, hidden)
            # delta_W shape: (out_features, in_features)
            # We need to apply: output += input @ delta_W.T
            x = input[0]  # (batch, seq, in_features)
            delta_out = F.linear(x.to(torch.bfloat16), delta_W)
            return output + delta_out

        return hook

    def register_hooks(self):
        """Register forward hooks for all LoRA layers."""
        self.remove_hooks()

        for name, module in self.base_model.named_modules():
            for layer in self.share.layer_names:
                if name == layer.replace("base_model.model.", ""):
                    hook = self._create_forward_hook(layer)
                    handle = module.register_forward_hook(hook)
                    self.hooks.append(handle)

        logger.info(f"Registered {len(self.hooks)} LoRA hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def train(
        self,
        train_data: List[Dict],
        task_id: str,
        steps: int = None,
        lr: float = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Train coefficients on new task data.

        This is Phase 2 of Share:
        - Basis is FROZEN
        - Only coefficients are trained
        - Much fewer parameters = less forgetting
        """
        steps = steps or self.config.training_steps
        lr = lr or self.config.learning_rate

        logger.info("=" * 60)
        logger.info("Phase 2: Coefficient-Only Training")
        logger.info("=" * 60)
        logger.info(f"Task: {task_id}")
        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Steps: {steps}, LR: {lr}")

        # Initialize fresh coefficients
        total_params = self.initialize_coefficients()

        # Create optimizer for coefficients ONLY
        params = list(self.eps_beta.values()) + list(self.eps_alpha.values())
        optimizer = torch.optim.AdamW(params, lr=lr)

        # Register hooks
        self.register_hooks()
        self.training = True

        # Training loop
        losses = []
        for step in range(steps):
            # Sample a batch
            idx = step % len(train_data)
            example = train_data[idx]

            # Tokenize
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(self.device)

            # Forward pass (hooks add delta_W)
            outputs = self.base_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=tokens.input_ids,
            )

            loss = outputs.loss

            # Backward pass (only affects coefficients)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (step + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                logger.info(f"Step {step+1}/{steps} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

        self.training = False
        self.remove_hooks()

        # Extract trained coefficients
        coefficients = {}
        for layer in self.share.layer_names:
            coefficients[layer] = (
                self.eps_beta[layer].detach().cpu().numpy().astype(np.float32),
                self.eps_alpha[layer].detach().cpu().numpy().astype(np.float32),
            )

        logger.info(f"Phase 2 complete. Final loss: {np.mean(losses[-10:]):.4f}")

        return coefficients


def phase2_train(
    share_model: ShareModel,
    train_data: List[Dict],
    task_id: str,
    config: ShareConfig = None,
) -> ShareModel:
    """
    Phase 2: Train coefficients for a new task.

    The basis is FROZEN - only tiny coefficient matrices are updated.
    This is what prevents catastrophic forgetting.
    """
    config = config or share_model.config

    trainer = CoefficientTrainer(
        share_model,
        config.base_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    coefficients = trainer.train(
        train_data,
        task_id,
        steps=config.training_steps,
        lr=config.learning_rate,
    )

    # Add to model
    share_model.coefficients[task_id] = coefficients
    share_model.task_ids.append(task_id)

    return share_model


# ==============================================================================
# Phase 3: Merge and Update Basis
# ==============================================================================

def phase3_merge(
    share_model: ShareModel,
    new_adapters: List[Path] = None,
) -> ShareModel:
    """
    Phase 3: Merge new knowledge and update basis.

    From paper:
    1. Reconstruct all task adapters from current basis + coefficients
    2. Add any new raw adapters
    3. Re-run SVD to extract updated principal factors
    4. Recalculate all coefficients analytically
    """
    logger.info("=" * 60)
    logger.info("Phase 3: Merge and Update Basis")
    logger.info("=" * 60)

    config = share_model.config
    p = config.pseudo_rank

    # Reconstruct all existing task adapters
    reconstructed_adapters = []
    for task_id in share_model.task_ids:
        adapter = {}
        for layer in share_model.layer_names:
            eps_beta, eps_alpha = share_model.coefficients[task_id][layer]
            beta = share_model.beta[layer]
            alpha = share_model.alpha[layer]

            B_hat = beta @ eps_beta  # (n, p)
            A_hat = (alpha @ eps_alpha).T  # (p, d)

            adapter[f"{layer}.lora_B.weight"] = torch.from_numpy(B_hat)
            adapter[f"{layer}.lora_A.weight"] = torch.from_numpy(A_hat)

        reconstructed_adapters.append(adapter)

    # Add new raw adapters if any
    if new_adapters:
        for adapter_dir in new_adapters:
            adapter = load_file(adapter_dir / "adapter_model.safetensors")
            reconstructed_adapters.append(adapter)

    logger.info(f"Merging {len(reconstructed_adapters)} adapters")

    # Re-run Phase 1 with all adapters
    new_model = ShareModel(config)
    new_model.layer_names = share_model.layer_names
    new_model.r = share_model.r

    for layer in share_model.layer_names:
        # Stack all B and A matrices
        B_list = []
        A_list = []
        for adapter in reconstructed_adapters:
            B = adapter[f"{layer}.lora_B.weight"].float().numpy()
            A = adapter[f"{layer}.lora_A.weight"].float().numpy()
            B_list.append(B)
            A_list.append(A.T)  # Transpose to get columns

        B_stack = np.concatenate(B_list, axis=1)  # (n, T*p)
        A_stack = np.concatenate(A_list, axis=1)  # (d, T*p)

        # Center
        B_centered = B_stack - B_stack.mean(axis=1, keepdims=True)
        A_centered = A_stack - A_stack.mean(axis=1, keepdims=True)

        # SVD
        U_b, S_b, _ = np.linalg.svd(B_centered, full_matrices=False)
        U_a, S_a, _ = np.linalg.svd(A_centered, full_matrices=False)

        # Select k
        var_b = (S_b ** 2).cumsum() / (S_b ** 2).sum()
        var_a = (S_a ** 2).cumsum() / (S_a ** 2).sum()
        k_beta = int(np.searchsorted(var_b, config.variance_threshold) + 1)
        k_alpha = int(np.searchsorted(var_a, config.variance_threshold) + 1)
        k_beta = max(p, min(k_beta, len(S_b)))
        k_alpha = max(p, min(k_alpha, len(S_a)))

        new_model.beta[layer] = U_b[:, :k_beta].astype(np.float32)
        new_model.alpha[layer] = U_a[:, :k_alpha].astype(np.float32)
        new_model.k_beta[layer] = k_beta
        new_model.k_alpha[layer] = k_alpha

    # Recalculate coefficients for all tasks
    for i, (task_id, adapter) in enumerate(zip(share_model.task_ids, reconstructed_adapters)):
        new_model.task_ids.append(task_id)
        new_model.coefficients[task_id] = {}

        for layer in new_model.layer_names:
            B = adapter[f"{layer}.lora_B.weight"].float().numpy()
            A = adapter[f"{layer}.lora_A.weight"].float().numpy()

            beta = new_model.beta[layer]
            alpha = new_model.alpha[layer]

            eps_beta = (beta.T @ B)[:, :p]
            eps_alpha = (alpha.T @ A.T)[:, :p]

            new_model.coefficients[task_id][layer] = (
                eps_beta.astype(np.float32),
                eps_alpha.astype(np.float32),
            )

    logger.info("Phase 3 complete")
    sample = new_model.layer_names[0]
    old_k = (share_model.k_beta.get(sample, 0), share_model.k_alpha.get(sample, 0))
    new_k = (new_model.k_beta[sample], new_model.k_alpha[sample])
    logger.info(f"  k changed: {old_k} -> {new_k}")

    return new_model


# ==============================================================================
# Inference: Export Adapter for Specific Task or General Use
# ==============================================================================

def export_task_adapter(
    share_model: ShareModel,
    task_id: str,
    output_dir: Path,
):
    """Export LoRA adapter for a specific task."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = share_model.reconstruct_lora(task_id)
    save_file(tensors, output_dir / "adapter_model.safetensors")

    # Save adapter config
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": share_model.config.pseudo_rank,
        "lora_alpha": share_model.config.pseudo_rank,
        "target_modules": list(set(
            layer.split(".")[-1] for layer in share_model.layer_names
        )),
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Exported adapter for task '{task_id}' to {output_dir}")


def export_general_adapter(
    share_model: ShareModel,
    output_dir: Path,
    method: str = "average",
):
    """
    Export a general-purpose adapter.

    Methods:
    - "average": Average coefficients across all tasks
    - "latest": Use most recent task's coefficients
    - "weighted": Weight by task recency/importance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "average":
        # Average coefficients across tasks
        avg_coefficients = {}
        for layer in share_model.layer_names:
            eps_betas = [share_model.coefficients[t][layer][0] for t in share_model.task_ids]
            eps_alphas = [share_model.coefficients[t][layer][1] for t in share_model.task_ids]

            avg_coefficients[layer] = (
                np.mean(eps_betas, axis=0),
                np.mean(eps_alphas, axis=0),
            )

        # Reconstruct adapter
        tensors = {}
        for layer in share_model.layer_names:
            eps_beta, eps_alpha = avg_coefficients[layer]
            beta = share_model.beta[layer]
            alpha = share_model.alpha[layer]

            B_hat = beta @ eps_beta
            A_hat = (alpha @ eps_alpha).T

            tensors[f"{layer}.lora_B.weight"] = torch.from_numpy(B_hat).to(torch.bfloat16)
            tensors[f"{layer}.lora_A.weight"] = torch.from_numpy(A_hat).to(torch.bfloat16)

    elif method == "latest":
        task_id = share_model.task_ids[-1]
        tensors = share_model.reconstruct_lora(task_id)

    else:
        raise ValueError(f"Unknown method: {method}")

    save_file(tensors, output_dir / "adapter_model.safetensors")

    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": share_model.config.pseudo_rank,
        "lora_alpha": share_model.config.pseudo_rank,
        "target_modules": list(set(
            layer.split(".")[-1] for layer in share_model.layer_names
        )),
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Exported general adapter ({method}) to {output_dir}")


# ==============================================================================
# Main CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete Share Algorithm (arXiv:2602.06043 + UWSH concepts)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Phase 1: Initialize
    p1 = subparsers.add_parser("phase1", help="Initialize shared basis from adapters")
    p1.add_argument("--adapters", "-a", required=True, help="Directory containing adapter subdirs")
    p1.add_argument("--output", "-o", required=True, help="Output Share model directory")
    p1.add_argument("--variance", "-v", type=float, default=0.6, help="Variance threshold for k")
    p1.add_argument("--pseudo-rank", "-p", type=int, default=1, help="Pseudo-rank p")

    # Phase 2: Train coefficients
    p2 = subparsers.add_parser("phase2", help="Train coefficients for new task")
    p2.add_argument("--share", "-s", required=True, help="Share model directory")
    p2.add_argument("--data", "-d", required=True, help="Training data JSONL")
    p2.add_argument("--task-id", "-t", required=True, help="Task identifier")
    p2.add_argument("--output", "-o", required=True, help="Output Share model directory")
    p2.add_argument("--steps", type=int, default=100, help="Training steps")
    p2.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Phase 3: Merge
    p3 = subparsers.add_parser("phase3", help="Merge and update basis")
    p3.add_argument("--share", "-s", required=True, help="Share model directory")
    p3.add_argument("--output", "-o", required=True, help="Output Share model directory")
    p3.add_argument("--new-adapters", "-n", nargs="*", help="New adapter directories to add")

    # Export
    exp = subparsers.add_parser("export", help="Export adapter for inference")
    exp.add_argument("--share", "-s", required=True, help="Share model directory")
    exp.add_argument("--output", "-o", required=True, help="Output adapter directory")
    exp.add_argument("--task", "-t", help="Specific task ID (or 'average'/'latest')")

    args = parser.parse_args()

    if args.command == "phase1":
        config = ShareConfig(
            variance_threshold=args.variance,
            pseudo_rank=args.pseudo_rank,
        )

        # Find adapter directories
        adapters_dir = Path(args.adapters)
        adapter_dirs = [
            d for d in adapters_dir.iterdir()
            if d.is_dir() or d.is_symlink()
        ]
        # Resolve symlinks
        adapter_dirs = [d.resolve() if d.is_symlink() else d for d in adapter_dirs]

        model = phase1_initialize(adapter_dirs, config)
        model.save(Path(args.output))

    elif args.command == "phase2":
        model = ShareModel()
        model.load(Path(args.share))
        model.config.training_steps = args.steps
        model.config.learning_rate = args.lr

        # Load training data
        with open(args.data) as f:
            train_data = [json.loads(line) for line in f]

        model = phase2_train(model, train_data, args.task_id)
        model.save(Path(args.output))

    elif args.command == "phase3":
        model = ShareModel()
        model.load(Path(args.share))

        new_adapters = [Path(a) for a in args.new_adapters] if args.new_adapters else None
        model = phase3_merge(model, new_adapters)
        model.save(Path(args.output))

    elif args.command == "export":
        model = ShareModel()
        model.load(Path(args.share))

        if args.task in ["average", "latest"]:
            export_general_adapter(model, Path(args.output), method=args.task)
        else:
            export_task_adapter(model, args.task, Path(args.output))


if __name__ == "__main__":
    main()
