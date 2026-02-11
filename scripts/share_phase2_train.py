#!/usr/bin/env python3
"""
Share Phase 2: Train coefficients only with frozen basis.

This is the KEY to preventing catastrophic forgetting per arXiv:2602.06043:
- Basis vectors (β, α) are FROZEN
- Only coefficients (ε_β, ε_α) are trained
- Coefficients initialized to ZERO (= no change from base model)
- After training, exports as standard LoRA adapter for compatibility

This trains ~460x fewer parameters than full LoRA!
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"


class ShareCoefficients(nn.Module):
    """Trainable Share coefficients with frozen basis."""

    def __init__(self, share_dir: Path, init_zero: bool = True):
        super().__init__()

        with open(share_dir / "metadata.json") as f:
            metadata = json.load(f)

        self.layer_names = metadata["layer_names"]

        # Get r (LoRA rank) from first coefficient in Share model
        coef_dir = share_dir / "coefficients"
        first_task = sorted(coef_dir.iterdir())[0]
        safe_first = self.layer_names[0].replace(".", "_")
        sample_eps = np.load(first_task / f"eps_beta_{safe_first}.npy")
        self.r = sample_eps.shape[1]

        # Load frozen basis vectors and track per-layer k values
        # Note: beta and alpha can have different k due to independent SVD
        basis_dir = share_dir / "basis"
        self.betas = {}
        self.alphas = {}
        self.layer_k_beta = {}
        self.layer_k_alpha = {}

        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            beta = np.load(basis_dir / f"beta_{safe_name}.npy")
            alpha = np.load(basis_dir / f"alpha_{safe_name}.npy")

            # Store per-layer k for each (may differ between beta and alpha)
            self.layer_k_beta[layer_name] = beta.shape[1]
            self.layer_k_alpha[layer_name] = alpha.shape[1]

            # Register as non-trainable buffers
            self.register_buffer(f"beta_{safe_name}", torch.from_numpy(beta).float())
            self.register_buffer(f"alpha_{safe_name}", torch.from_numpy(alpha).float())
            self.betas[layer_name] = f"beta_{safe_name}"
            self.alphas[layer_name] = f"alpha_{safe_name}"

        # Initialize trainable coefficients with per-layer k × r
        # CRITICAL: Zero init = base model behavior
        self.eps_beta = nn.ParameterDict()
        self.eps_alpha = nn.ParameterDict()

        total_params = 0
        for layer_name in self.layer_names:
            safe_name = layer_name.replace(".", "_")
            k_beta = self.layer_k_beta[layer_name]
            k_alpha = self.layer_k_alpha[layer_name]
            if init_zero:
                self.eps_beta[safe_name] = nn.Parameter(torch.zeros(k_beta, self.r))
                self.eps_alpha[safe_name] = nn.Parameter(torch.zeros(k_alpha, self.r))
            else:
                self.eps_beta[safe_name] = nn.Parameter(torch.randn(k_beta, self.r) * 0.01)
                self.eps_alpha[safe_name] = nn.Parameter(torch.randn(k_alpha, self.r) * 0.01)
            total_params += (k_beta + k_alpha) * self.r

        # Summary statistics
        k_beta_values = list(self.layer_k_beta.values())
        k_alpha_values = list(self.layer_k_alpha.values())
        logger.info(f"Share Phase 2: r={self.r}, layers={len(self.layer_names)}")
        logger.info(f"k_beta: min={min(k_beta_values)}, max={max(k_beta_values)}")
        logger.info(f"k_alpha: min={min(k_alpha_values)}, max={max(k_alpha_values)}")
        logger.info(f"Total trainable parameters: {total_params:,}")

    def get_lora_matrices(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct B and A matrices for a layer.

        B_hat = β @ ε_β  →  (n, k) @ (k, r) = (n, r)
        A_hat = (α @ ε_α).T  →  ((d, k) @ (k, r)).T = (r, d)
        """
        safe_name = layer_name.replace(".", "_")

        beta = getattr(self, self.betas[layer_name])
        alpha = getattr(self, self.alphas[layer_name])

        eps_beta = self.eps_beta[safe_name]
        eps_alpha = self.eps_alpha[safe_name]

        B_hat = beta @ eps_beta  # (n, r)
        A_hat = (alpha @ eps_alpha).T  # (r, d)

        return B_hat, A_hat

    def export_lora_adapter(self, output_path: Path, original_config_path: Path = None):
        """Export trained coefficients as standard LoRA adapter."""
        tensors = {}

        for layer_name in self.layer_names:
            B_hat, A_hat = self.get_lora_matrices(layer_name)
            tensors[f"{layer_name}.lora_B.weight"] = B_hat.detach().to(torch.bfloat16).contiguous()
            tensors[f"{layer_name}.lora_A.weight"] = A_hat.detach().to(torch.bfloat16).contiguous()

        output_path.mkdir(parents=True, exist_ok=True)
        save_file(tensors, output_path / "adapter_model.safetensors")

        # Create adapter_config.json
        config = {
            "peft_type": "LORA",
            "base_model_name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "task_type": "CAUSAL_LM",
            "r": self.r,
            "lora_alpha": self.r,  # Scale 1.0
            "lora_dropout": 0.0,
            "target_modules": list(set(n.split(".")[-1] for n in self.layer_names)),
            "bias": "none",
        }

        with open(output_path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Exported LoRA adapter to {output_path}")


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

        # Create labels (shift for causal LM)
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def strip_peft_prefix(layer_name: str) -> str:
    """Strip PEFT prefix from layer name (e.g., 'base_model.model.model.layers...' -> 'model.layers...')."""
    # PEFT adds 'base_model.model.' prefix
    prefixes = ["base_model.model.", "base_model."]
    for prefix in prefixes:
        if layer_name.startswith(prefix):
            return layer_name[len(prefix):]
    return layer_name


def create_share_lora_model(base_model, share_coefs: ShareCoefficients):
    """
    Create a model that applies Share deltas during forward pass.

    We hook into the linear layers and add the LoRA delta.
    """
    hooks = []

    # Build mapping from base model layer names to Share layer names
    layer_mapping = {}
    for share_name in share_coefs.layer_names:
        base_name = strip_peft_prefix(share_name)
        layer_mapping[base_name] = share_name

    def make_hook(share_layer_name):
        def hook(module, input, output):
            B_hat, A_hat = share_coefs.get_lora_matrices(share_layer_name)
            B_hat = B_hat.to(output.device, output.dtype)
            A_hat = A_hat.to(output.device, output.dtype)

            x = input[0] if isinstance(input, tuple) else input
            # LoRA: output += x @ A.T @ B.T = x @ (B @ A).T
            delta = x @ A_hat.T @ B_hat.T
            return output + delta
        return hook

    # Register hooks on target modules
    for name, module in base_model.named_modules():
        if name in layer_mapping:
            share_name = layer_mapping[name]
            h = module.register_forward_hook(make_hook(share_name))
            hooks.append(h)

    logger.info(f"Layer mapping: {len(layer_mapping)} Share layers -> {len(hooks)} hooks registered")
    return hooks


def train_phase2(
    share_dir: Path,
    data_path: Path,
    output_dir: Path,
    base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    steps: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    init_zero: bool = True,
):
    """Train Share Phase 2 (coefficients only)."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Share Phase 2: Coefficient-Only Training")
    logger.info("=" * 50)
    logger.info(f"Share basis: {share_dir}")
    logger.info(f"Training data: {data_path}")
    logger.info(f"Output: {run_dir}")
    logger.info(f"Steps: {steps}, LR: {lr}, Batch: {batch_size}")
    logger.info(f"Init zero: {init_zero}")

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

    # Initialize Share coefficients
    share_coefs = ShareCoefficients(share_dir, init_zero=init_zero)
    share_coefs = share_coefs.cuda().float()
    share_coefs.train()

    # Register hooks
    hooks = create_share_lora_model(model, share_coefs)
    logger.info(f"Registered {len(hooks)} LoRA hooks")

    # Create dataset and dataloader
    dataset = SFTDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer (only coefficients)
    optimizer = torch.optim.AdamW(share_coefs.parameters(), lr=lr, weight_decay=0.01)

    # Training loop
    step = 0
    total_loss = 0.0
    log_interval = max(1, steps // 10)

    logger.info("Starting training...")

    # Debug: verify coefficients require grad
    sample_param = list(share_coefs.parameters())[0]
    logger.info(f"Coefficient requires_grad: {sample_param.requires_grad}, device: {sample_param.device}")

    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break

            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            optimizer.zero_grad()

            # Forward pass (hooks apply Share deltas)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if step == 0:
                logger.info(f"Initial loss: {loss.item():.4f}, requires_grad: {loss.requires_grad}")

            # Backward
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

    # Export as LoRA adapter
    adapter_path = run_dir / "adapter"
    share_coefs.export_lora_adapter(adapter_path)

    # Save training info
    k_beta_values = list(share_coefs.layer_k_beta.values())
    k_alpha_values = list(share_coefs.layer_k_alpha.values())
    total_trainable = sum(
        (share_coefs.layer_k_beta[ln] + share_coefs.layer_k_alpha[ln]) * share_coefs.r
        for ln in share_coefs.layer_names
    )
    info = {
        "share_dir": str(share_dir),
        "data_path": str(data_path),
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "init_zero": init_zero,
        "final_loss": total_loss / steps,
        "k_beta_range": [min(k_beta_values), max(k_beta_values)],
        "k_alpha_range": [min(k_alpha_values), max(k_alpha_values)],
        "r": share_coefs.r,
        "trainable_params": total_trainable,
    }
    with open(run_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Training complete! Adapter saved to: {adapter_path}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Share Phase 2: Train coefficients only")
    parser.add_argument("--share", "-s", required=True, help="Path to Share model directory")
    parser.add_argument("--data", "-d", required=True, help="Training data (JSONL)")
    parser.add_argument("--output", "-o", default="runs/adapters/phase2", help="Output directory")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--no-init-zero", action="store_true", help="Random init instead of zero")
    args = parser.parse_args()

    train_phase2(
        share_dir=Path(args.share),
        data_path=Path(args.data),
        output_dir=Path(args.output),
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        init_zero=not args.no_init_zero,
    )


if __name__ == "__main__":
    main()
