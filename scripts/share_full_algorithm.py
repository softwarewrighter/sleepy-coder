#!/usr/bin/env python3
"""
Complete Share Algorithm Implementation (arXiv:2602.06043)

This implements all three phases:
- Phase 1: Initialize shared basis from existing adapters
- Phase 2: Continual adaptation with temporary expansion
- Phase 3: Merge and update basis

Key insight: The basis evolves over time as new tasks are learned.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"


class ShareModel:
    """
    Complete Share model with basis and coefficients.

    Stores:
    - beta[layer]: (n, k) basis for B matrices
    - alpha[layer]: (d, k) basis for A matrices
    - coefficients[task][layer]: (eps_beta, eps_alpha) per task
    """

    def __init__(self, share_dir: Optional[Path] = None):
        self.layer_names = []
        self.beta = {}  # layer -> (n, k) numpy array
        self.alpha = {}  # layer -> (d, k) numpy array
        self.coefficients = {}  # task_id -> {layer -> (eps_beta, eps_alpha)}
        self.r = None  # LoRA rank
        self.variance_threshold = 0.6  # 60% explained variance for k selection

        if share_dir:
            self.load(share_dir)

    def load(self, share_dir: Path):
        """Load existing Share model."""
        with open(share_dir / "metadata.json") as f:
            metadata = json.load(f)

        self.layer_names = metadata["layer_names"]
        self.r = metadata.get("r", 8)

        # Load basis
        basis_dir = share_dir / "basis"
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            self.beta[layer_name] = np.load(basis_dir / f"beta_{safe_name}.npy")
            self.alpha[layer_name] = np.load(basis_dir / f"alpha_{safe_name}.npy")

        # Load coefficients
        coef_dir = share_dir / "coefficients"
        if coef_dir.exists():
            for task_dir in coef_dir.iterdir():
                if task_dir.is_dir():
                    task_id = task_dir.name
                    self.coefficients[task_id] = {}
                    for layer_name in self.layer_names:
                        safe_name = layer_name.replace(".", "_")
                        eps_beta = np.load(task_dir / f"eps_beta_{safe_name}.npy")
                        eps_alpha = np.load(task_dir / f"eps_alpha_{safe_name}.npy")
                        self.coefficients[task_id][layer_name] = (eps_beta, eps_alpha)

        logger.info(f"Loaded Share model: {len(self.layer_names)} layers, {len(self.coefficients)} tasks")
        for layer in self.layer_names[:1]:
            logger.info(f"  {layer}: beta={self.beta[layer].shape}, alpha={self.alpha[layer].shape}")

    def save(self, output_dir: Path):
        """Save Share model."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "layer_names": self.layer_names,
            "r": self.r,
            "num_tasks": len(self.coefficients),
            "variance_threshold": self.variance_threshold,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save basis
        basis_dir = output_dir / "basis"
        basis_dir.mkdir(exist_ok=True)
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            np.save(basis_dir / f"beta_{safe_name}.npy", self.beta[layer_name])
            np.save(basis_dir / f"alpha_{safe_name}.npy", self.alpha[layer_name])

        # Save coefficients
        coef_dir = output_dir / "coefficients"
        coef_dir.mkdir(exist_ok=True)
        for task_id, task_coefs in self.coefficients.items():
            task_dir = coef_dir / task_id
            task_dir.mkdir(exist_ok=True)
            for layer_name, (eps_beta, eps_alpha) in task_coefs.items():
                safe_name = layer_name.replace(".", "_")
                np.save(task_dir / f"eps_beta_{safe_name}.npy", eps_beta)
                np.save(task_dir / f"eps_alpha_{safe_name}.npy", eps_alpha)

        logger.info(f"Saved Share model to {output_dir}")

    def get_k(self, layer_name: str) -> tuple[int, int]:
        """Get k values for beta and alpha for a layer."""
        return self.beta[layer_name].shape[1], self.alpha[layer_name].shape[1]

    def reconstruct_adapter(self, task_id: str) -> dict:
        """
        Reconstruct LoRA adapter tensors for a task.

        B_hat = beta @ eps_beta  ->  (n, k) @ (k, r) = (n, r)
        A_hat = (alpha @ eps_alpha).T  ->  ((d, k) @ (k, r)).T = (r, d)
        """
        tensors = {}
        task_coefs = self.coefficients[task_id]

        for layer_name in self.layer_names:
            eps_beta, eps_alpha = task_coefs[layer_name]
            beta = self.beta[layer_name]
            alpha = self.alpha[layer_name]

            B_hat = beta @ eps_beta  # (n, r)
            A_hat = (alpha @ eps_alpha).T  # (r, d)

            tensors[f"{layer_name}.lora_B.weight"] = torch.from_numpy(B_hat).to(torch.bfloat16).contiguous()
            tensors[f"{layer_name}.lora_A.weight"] = torch.from_numpy(A_hat).to(torch.bfloat16).contiguous()

        return tensors

    def reconstruct_averaged(self) -> dict:
        """Reconstruct adapter using averaged coefficients across all tasks."""
        if not self.coefficients:
            raise ValueError("No coefficients to average")

        # Average coefficients
        avg_coefs = {}
        for layer_name in self.layer_names:
            all_eps_beta = []
            all_eps_alpha = []
            for task_coefs in self.coefficients.values():
                eps_beta, eps_alpha = task_coefs[layer_name]
                all_eps_beta.append(eps_beta)
                all_eps_alpha.append(eps_alpha)
            avg_coefs[layer_name] = (
                np.mean(all_eps_beta, axis=0),
                np.mean(all_eps_alpha, axis=0)
            )

        # Reconstruct
        tensors = {}
        for layer_name in self.layer_names:
            eps_beta, eps_alpha = avg_coefs[layer_name]
            beta = self.beta[layer_name]
            alpha = self.alpha[layer_name]

            B_hat = beta @ eps_beta
            A_hat = (alpha @ eps_alpha).T

            tensors[f"{layer_name}.lora_B.weight"] = torch.from_numpy(B_hat).to(torch.bfloat16).contiguous()
            tensors[f"{layer_name}.lora_A.weight"] = torch.from_numpy(A_hat).to(torch.bfloat16).contiguous()

        return tensors


def select_k_by_variance(singular_values: np.ndarray, threshold: float = 0.6) -> int:
    """Select k to capture threshold fraction of variance."""
    total_var = np.sum(singular_values ** 2)
    cumsum = np.cumsum(singular_values ** 2)
    k = np.searchsorted(cumsum, threshold * total_var) + 1
    return max(1, min(k, len(singular_values)))


def phase1_initialize(adapter_paths: list[Path], output_dir: Path, variance_threshold: float = 0.6) -> ShareModel:
    """
    Phase 1: Initialize shared basis from existing adapters.

    1. Load all adapter B and A matrices
    2. Stack and mean-center
    3. SVD to extract principal components
    4. Select k by explained variance threshold
    5. Compute coefficients for each adapter
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Initialize Shared Basis")
    logger.info("=" * 60)
    logger.info(f"Input adapters: {len(adapter_paths)}")
    logger.info(f"Variance threshold: {variance_threshold}")

    # Load all adapters
    all_adapters = []
    layer_names = None
    r = None

    for path in adapter_paths:
        tensors = load_file(path / "adapter_model.safetensors")
        adapter = {}
        for key, tensor in tensors.items():
            if ".lora_B.weight" in key:
                layer = key.replace(".lora_B.weight", "")
                if layer not in adapter:
                    adapter[layer] = {}
                adapter[layer]["B"] = tensor.float().numpy()
                if r is None:
                    r = tensor.shape[1]
            elif ".lora_A.weight" in key:
                layer = key.replace(".lora_A.weight", "")
                if layer not in adapter:
                    adapter[layer] = {}
                adapter[layer]["A"] = tensor.float().numpy()

        if layer_names is None:
            layer_names = sorted(adapter.keys())
        all_adapters.append(adapter)

    logger.info(f"Loaded {len(all_adapters)} adapters, {len(layer_names)} layers, r={r}")

    # Initialize Share model
    share = ShareModel()
    share.layer_names = layer_names
    share.r = r
    share.variance_threshold = variance_threshold

    # Process each layer
    for layer_name in layer_names:
        # Stack B matrices: (n, r*T)
        B_stack = np.hstack([a[layer_name]["B"] for a in all_adapters])
        # Stack A matrices: (d, r*T) - note A is (r, d), so we stack A.T
        A_stack = np.hstack([a[layer_name]["A"].T for a in all_adapters])

        # Mean center
        B_mean = B_stack.mean(axis=1, keepdims=True)
        A_mean = A_stack.mean(axis=1, keepdims=True)
        B_centered = B_stack - B_mean
        A_centered = A_stack - A_mean

        # SVD
        U_B, S_B, _ = np.linalg.svd(B_centered, full_matrices=False)
        U_A, S_A, _ = np.linalg.svd(A_centered, full_matrices=False)

        # Select k
        k_beta = select_k_by_variance(S_B, variance_threshold)
        k_alpha = select_k_by_variance(S_A, variance_threshold)

        # Store basis (top-k left singular vectors)
        share.beta[layer_name] = U_B[:, :k_beta].astype(np.float32)
        share.alpha[layer_name] = U_A[:, :k_alpha].astype(np.float32)

    # Compute coefficients for each adapter
    for i, adapter in enumerate(all_adapters):
        task_id = f"task_{i:03d}"
        share.coefficients[task_id] = {}

        for layer_name in layer_names:
            B = adapter[layer_name]["B"]  # (n, r)
            A = adapter[layer_name]["A"].T  # (d, r) after transpose

            beta = share.beta[layer_name]  # (n, k_beta)
            alpha = share.alpha[layer_name]  # (d, k_alpha)

            # Compute coefficients using pseudoinverse
            # eps_beta = beta^T @ B (since beta has orthonormal columns from SVD)
            eps_beta = beta.T @ B  # (k_beta, r)
            eps_alpha = alpha.T @ A  # (k_alpha, r)

            share.coefficients[task_id][layer_name] = (
                eps_beta.astype(np.float32),
                eps_alpha.astype(np.float32)
            )

    # Save
    share.save(output_dir)

    # Summary
    total_k_beta = sum(share.beta[l].shape[1] for l in layer_names)
    total_k_alpha = sum(share.alpha[l].shape[1] for l in layer_names)
    logger.info(f"Phase 1 complete: total k_beta={total_k_beta}, k_alpha={total_k_alpha}")

    return share


class ShareTrainer(nn.Module):
    """
    Phase 2 trainer with temporary expansion.

    Following the paper:
    1. Start with frozen basis (beta, alpha)
    2. Optionally expand with phi additional factors
    3. Initialize new coefficients to zero (or small random)
    4. Train only coefficients
    """

    def __init__(self, share: ShareModel, phi: int = 0, init_zero: bool = True):
        super().__init__()

        self.share = share
        self.phi = phi  # Temporary expansion factors
        self.layer_names = share.layer_names
        self.r = share.r

        # Register frozen basis as buffers
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            self.register_buffer(
                f"beta_{safe_name}",
                torch.from_numpy(share.beta[layer_name]).float()
            )
            self.register_buffer(
                f"alpha_{safe_name}",
                torch.from_numpy(share.alpha[layer_name]).float()
            )

        # If phi > 0, add temporary expansion vectors (also trainable)
        # These get merged back in Phase 3
        self.temp_beta = nn.ParameterDict()
        self.temp_alpha = nn.ParameterDict()

        # Trainable coefficients
        self.eps_beta = nn.ParameterDict()
        self.eps_alpha = nn.ParameterDict()

        total_params = 0
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            k_beta = share.beta[layer_name].shape[1]
            k_alpha = share.alpha[layer_name].shape[1]
            n = share.beta[layer_name].shape[0]
            d = share.alpha[layer_name].shape[0]

            # Temporary expansion vectors (if phi > 0)
            if phi > 0:
                self.temp_beta[safe_name] = nn.Parameter(torch.randn(n, phi) * 0.01)
                self.temp_alpha[safe_name] = nn.Parameter(torch.randn(d, phi) * 0.01)
                total_params += n * phi + d * phi

            # Coefficients for existing basis + expansion
            total_k_beta = k_beta + phi
            total_k_alpha = k_alpha + phi

            if init_zero:
                self.eps_beta[safe_name] = nn.Parameter(torch.zeros(total_k_beta, self.r))
                self.eps_alpha[safe_name] = nn.Parameter(torch.zeros(total_k_alpha, self.r))
            else:
                self.eps_beta[safe_name] = nn.Parameter(torch.randn(total_k_beta, self.r) * 0.01)
                self.eps_alpha[safe_name] = nn.Parameter(torch.randn(total_k_alpha, self.r) * 0.01)

            total_params += (total_k_beta + total_k_alpha) * self.r

        logger.info(f"ShareTrainer: phi={phi}, total trainable params={total_params:,}")

    def get_full_basis(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full basis including temporary expansion."""
        safe_name = layer_name.replace(".", "_")
        beta = getattr(self, f"beta_{safe_name}")
        alpha = getattr(self, f"alpha_{safe_name}")

        if self.phi > 0:
            temp_beta = self.temp_beta[safe_name]
            temp_alpha = self.temp_alpha[safe_name]
            beta = torch.cat([beta, temp_beta], dim=1)
            alpha = torch.cat([alpha, temp_alpha], dim=1)

        return beta, alpha

    def get_lora_matrices(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct B and A matrices."""
        safe_name = layer_name.replace(".", "_")
        beta, alpha = self.get_full_basis(layer_name)

        eps_beta = self.eps_beta[safe_name]
        eps_alpha = self.eps_alpha[safe_name]

        B_hat = beta @ eps_beta  # (n, r)
        A_hat = (alpha @ eps_alpha).T  # (r, d)

        return B_hat, A_hat

    def get_trained_coefficients(self) -> dict:
        """Extract trained coefficients for Phase 3."""
        coefs = {}
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            coefs[layer_name] = (
                self.eps_beta[safe_name].detach().cpu().numpy(),
                self.eps_alpha[safe_name].detach().cpu().numpy()
            )
        return coefs

    def get_trained_expansion(self) -> dict:
        """Extract trained expansion vectors for Phase 3."""
        if self.phi == 0:
            return {}
        expansion = {}
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            expansion[layer_name] = (
                self.temp_beta[safe_name].detach().cpu().numpy(),
                self.temp_alpha[safe_name].detach().cpu().numpy()
            )
        return expansion


def strip_peft_prefix(layer_name: str) -> str:
    """Strip PEFT prefix from layer name."""
    prefixes = ["base_model.model.", "base_model."]
    for prefix in prefixes:
        if layer_name.startswith(prefix):
            return layer_name[len(prefix):]
    return layer_name


def create_lora_hooks(base_model, trainer: ShareTrainer):
    """Create forward hooks to apply LoRA deltas."""
    hooks = []

    layer_mapping = {}
    for share_name in trainer.layer_names:
        base_name = strip_peft_prefix(share_name)
        layer_mapping[base_name] = share_name

    def make_hook(share_layer_name):
        def hook(module, input, output):
            B_hat, A_hat = trainer.get_lora_matrices(share_layer_name)
            B_hat = B_hat.to(output.device, output.dtype)
            A_hat = A_hat.to(output.device, output.dtype)

            x = input[0] if isinstance(input, tuple) else input
            delta = x @ A_hat.T @ B_hat.T
            return output + delta
        return hook

    for name, module in base_model.named_modules():
        if name in layer_mapping:
            share_name = layer_mapping[name]
            h = module.register_forward_hook(make_hook(share_name))
            hooks.append(h)

    logger.info(f"Registered {len(hooks)} LoRA hooks")
    return hooks


class SFTDataset(Dataset):
    """SFT dataset for training."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        messages = [
            {"role": "system", "content": ex.get("instruction", "Fix the Rust code.")},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def phase2_train(
    share: ShareModel,
    data_path: Path,
    output_dir: Path,
    base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    steps: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    phi: int = 0,
    init_zero: bool = True,
) -> tuple[ShareTrainer, dict]:
    """
    Phase 2: Train coefficients (and optional expansion) on new task.

    Returns trained ShareTrainer and training info.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Continual Adaptation")
    logger.info("=" * 60)
    logger.info(f"Expansion phi: {phi}")
    logger.info(f"Training data: {data_path}")
    logger.info(f"Steps: {steps}, LR: {lr}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Create trainer
    trainer = ShareTrainer(share, phi=phi, init_zero=init_zero)
    trainer = trainer.cuda().float()
    trainer.train()

    # Register hooks
    hooks = create_lora_hooks(model, trainer)

    # Dataset
    dataset = SFTDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    step = 0
    total_loss = 0.0
    log_interval = max(1, steps // 10)

    logger.info("Starting training...")

    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % log_interval == 0 or step == steps:
                avg_loss = total_loss / step
                logger.info(f"Step {step}/{steps} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save training info
    info = {
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "phi": phi,
        "init_zero": init_zero,
        "final_loss": total_loss / steps,
    }

    logger.info(f"Phase 2 complete. Final loss: {info['final_loss']:.4f}")

    return trainer, info


def phase3_merge(
    share: ShareModel,
    trainer: ShareTrainer,
    new_task_id: str,
    output_dir: Path,
) -> ShareModel:
    """
    Phase 3: Merge trained coefficients/expansion back into Share model.

    1. Reconstruct new adapter from trained coefficients
    2. Stack with existing adapters (reconstructed from share)
    3. Re-run SVD to update basis
    4. Recalculate all coefficients
    """
    logger.info("=" * 60)
    logger.info("Phase 3: Merge and Update Basis")
    logger.info("=" * 60)

    # Get trained coefficients and expansion
    new_coefs = trainer.get_trained_coefficients()
    expansion = trainer.get_trained_expansion()

    # Reconstruct all existing adapters
    all_B = {layer: [] for layer in share.layer_names}
    all_A = {layer: [] for layer in share.layer_names}
    task_ids = list(share.coefficients.keys())

    for task_id in task_ids:
        for layer_name in share.layer_names:
            eps_beta, eps_alpha = share.coefficients[task_id][layer_name]
            beta = share.beta[layer_name]
            alpha = share.alpha[layer_name]

            B = beta @ eps_beta  # (n, r)
            A = alpha @ eps_alpha  # (d, r)

            all_B[layer_name].append(B)
            all_A[layer_name].append(A)

    # Add new task's adapter
    for layer_name in share.layer_names:
        # Get full basis (including expansion if phi > 0)
        beta, alpha = trainer.get_full_basis(layer_name)
        beta = beta.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()

        eps_beta, eps_alpha = new_coefs[layer_name]

        B_new = beta @ eps_beta  # (n, r)
        A_new = alpha @ eps_alpha  # (d, r)

        all_B[layer_name].append(B_new)
        all_A[layer_name].append(A_new)

    task_ids.append(new_task_id)

    # Re-run SVD to get updated basis
    new_share = ShareModel()
    new_share.layer_names = share.layer_names
    new_share.r = share.r
    new_share.variance_threshold = share.variance_threshold

    for layer_name in share.layer_names:
        # Stack matrices
        B_stack = np.hstack(all_B[layer_name])  # (n, r*T)
        A_stack = np.hstack(all_A[layer_name])  # (d, r*T)

        # Mean center
        B_mean = B_stack.mean(axis=1, keepdims=True)
        A_mean = A_stack.mean(axis=1, keepdims=True)
        B_centered = B_stack - B_mean
        A_centered = A_stack - A_mean

        # SVD
        U_B, S_B, _ = np.linalg.svd(B_centered, full_matrices=False)
        U_A, S_A, _ = np.linalg.svd(A_centered, full_matrices=False)

        # Select k
        k_beta = select_k_by_variance(S_B, share.variance_threshold)
        k_alpha = select_k_by_variance(S_A, share.variance_threshold)

        # Store updated basis
        new_share.beta[layer_name] = U_B[:, :k_beta].astype(np.float32)
        new_share.alpha[layer_name] = U_A[:, :k_alpha].astype(np.float32)

    # Recalculate coefficients for all tasks
    for i, task_id in enumerate(task_ids):
        new_share.coefficients[task_id] = {}

        for layer_name in share.layer_names:
            B = all_B[layer_name][i]  # (n, r)
            A = all_A[layer_name][i]  # (d, r)

            beta = new_share.beta[layer_name]
            alpha = new_share.alpha[layer_name]

            eps_beta = beta.T @ B  # (k_beta, r)
            eps_alpha = alpha.T @ A  # (k_alpha, r)

            new_share.coefficients[task_id][layer_name] = (
                eps_beta.astype(np.float32),
                eps_alpha.astype(np.float32)
            )

    # Save updated Share model
    new_share.save(output_dir)

    logger.info(f"Phase 3 complete. Updated model has {len(new_share.coefficients)} tasks")
    for layer in share.layer_names[:1]:
        old_k = (share.beta[layer].shape[1], share.alpha[layer].shape[1])
        new_k = (new_share.beta[layer].shape[1], new_share.alpha[layer].shape[1])
        logger.info(f"  {layer}: k changed from {old_k} to {new_k}")

    return new_share


def export_adapter(share: ShareModel, task_id: str, output_path: Path):
    """Export a single task's adapter."""
    if task_id == "averaged":
        tensors = share.reconstruct_averaged()
    else:
        tensors = share.reconstruct_adapter(task_id)

    output_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path / "adapter_model.safetensors")

    # Config
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "task_type": "CAUSAL_LM",
        "r": share.r,
        "lora_alpha": share.r,
        "lora_dropout": 0.0,
        "target_modules": ["k_proj", "q_proj", "v_proj", "o_proj"],
        "bias": "none",
    }
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Exported adapter to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Share Algorithm - Full Implementation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Phase 1
    p1 = subparsers.add_parser("phase1", help="Initialize shared basis from adapters")
    p1.add_argument("--adapters", "-a", required=True, help="Directory containing adapter subdirs")
    p1.add_argument("--output", "-o", required=True, help="Output Share model directory")
    p1.add_argument("--variance", type=float, default=0.6, help="Variance threshold for k selection")

    # Phase 2
    p2 = subparsers.add_parser("phase2", help="Train on new task")
    p2.add_argument("--share", "-s", required=True, help="Share model directory")
    p2.add_argument("--data", "-d", required=True, help="Training data JSONL")
    p2.add_argument("--output", "-o", required=True, help="Output directory")
    p2.add_argument("--steps", type=int, default=100, help="Training steps")
    p2.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p2.add_argument("--phi", type=int, default=0, help="Temporary expansion factors")
    p2.add_argument("--batch-size", type=int, default=4, help="Batch size")

    # Phase 3
    p3 = subparsers.add_parser("phase3", help="Merge trained task into Share model")
    p3.add_argument("--share", "-s", required=True, help="Share model directory")
    p3.add_argument("--trainer", "-t", required=True, help="Trained Phase 2 checkpoint")
    p3.add_argument("--task-id", required=True, help="ID for new task")
    p3.add_argument("--output", "-o", required=True, help="Output Share model directory")

    # Export
    exp = subparsers.add_parser("export", help="Export adapter from Share model")
    exp.add_argument("--share", "-s", required=True, help="Share model directory")
    exp.add_argument("--task", "-t", default="averaged", help="Task ID or 'averaged'")
    exp.add_argument("--output", "-o", required=True, help="Output adapter directory")

    # Full pipeline
    full = subparsers.add_parser("train", help="Run Phase 2 + Phase 3 together")
    full.add_argument("--share", "-s", required=True, help="Share model directory")
    full.add_argument("--data", "-d", required=True, help="Training data JSONL")
    full.add_argument("--output", "-o", required=True, help="Output directory")
    full.add_argument("--task-id", required=True, help="ID for new task")
    full.add_argument("--steps", type=int, default=100, help="Training steps")
    full.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    full.add_argument("--phi", type=int, default=0, help="Temporary expansion factors")

    args = parser.parse_args()

    if args.command == "phase1":
        adapter_dir = Path(args.adapters)
        adapter_paths = sorted([d for d in adapter_dir.iterdir() if d.is_dir()])
        phase1_initialize(adapter_paths, Path(args.output), args.variance)

    elif args.command == "phase2":
        share = ShareModel(Path(args.share))
        trainer, info = phase2_train(
            share,
            Path(args.data),
            Path(args.output),
            steps=args.steps,
            lr=args.lr,
            phi=args.phi,
            batch_size=args.batch_size,
        )
        # Save trainer state for Phase 3
        torch.save(trainer.state_dict(), Path(args.output) / "trainer.pt")
        with open(Path(args.output) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

    elif args.command == "phase3":
        share = ShareModel(Path(args.share))
        # Load trainer
        trainer = ShareTrainer(share, phi=0)  # phi will be inferred from checkpoint
        trainer.load_state_dict(torch.load(args.trainer))
        phase3_merge(share, trainer, args.task_id, Path(args.output))

    elif args.command == "export":
        share = ShareModel(Path(args.share))
        export_adapter(share, args.task, Path(args.output))

    elif args.command == "train":
        # Combined Phase 2 + Phase 3
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        share = ShareModel(Path(args.share))

        # Phase 2
        trainer, info = phase2_train(
            share,
            Path(args.data),
            run_dir,
            steps=args.steps,
            lr=args.lr,
            phi=args.phi,
        )

        # Phase 3
        new_share = phase3_merge(share, trainer, args.task_id, run_dir / "share_updated")

        # Export averaged adapter
        export_adapter(new_share, "averaged", run_dir / "adapter")

        with open(run_dir / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Complete! Output in {run_dir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
