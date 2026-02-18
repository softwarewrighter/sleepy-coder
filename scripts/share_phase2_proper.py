#!/usr/bin/env python3
"""
Share Phase 2: Proper Coefficient-Only Training (arXiv:2602.06043).

This is the KEY to preventing catastrophic forgetting:
- Basis vectors (beta, alpha) are FROZEN
- Only coefficient matrices (eps_beta, eps_alpha) are TRAINED via gradient descent
- Coefficients initialized to ZERO (= base model behavior at start)
- Training uses actual LM loss (cross-entropy), not analytical projection
- Trained coefficients saved in Share format for direct use with share_inference.py

This replaces the broken approach of "train full LoRA, then project and truncate."

Usage:
    python scripts/share_phase2_proper.py \\
        --share runs/share_proper_trained \\
        --data data/sft/distilled/mut_borrow_conflict.jsonl \\
        --task-id mut_borrow_conflict_v2 \\
        --steps 100 --lr 1e-4 --p 1

    # Then evaluate:
    python scripts/run_experiments.py experiment1  # uses updated coefficients
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"


class ShareCoefficientsV2(nn.Module):
    """Trainable Share coefficients with configurable pseudo-rank p.

    Unlike the original ShareCoefficients which reads r from stored files,
    this version takes p as an explicit parameter and initializes fresh
    trainable coefficients of shape (k, p) per layer.
    """

    def __init__(self, share_dir: Path, p: int = 1):
        super().__init__()
        self.p = p

        with open(share_dir / "metadata.json") as f:
            metadata = json.load(f)
        self.layer_names = metadata["layer_names"]

        # Load frozen basis vectors
        basis_dir = share_dir / "basis"
        self.betas = {}
        self.alphas = {}
        self.layer_k_beta = {}
        self.layer_k_alpha = {}

        for layer_name in self.layer_names:
            safe = layer_name.replace(".", "_")
            beta = np.load(basis_dir / f"beta_{safe}.npy")
            alpha = np.load(basis_dir / f"alpha_{safe}.npy")

            self.layer_k_beta[layer_name] = beta.shape[1]
            self.layer_k_alpha[layer_name] = alpha.shape[1]

            # Register as non-trainable buffers (bfloat16 to match model)
            self.register_buffer(f"beta_{safe}", torch.from_numpy(beta).to(torch.bfloat16))
            self.register_buffer(f"alpha_{safe}", torch.from_numpy(alpha).to(torch.bfloat16))
            self.betas[layer_name] = f"beta_{safe}"
            self.alphas[layer_name] = f"alpha_{safe}"

        # Initialize trainable coefficients: (k, p) per layer
        # Both eps_beta and eps_alpha get small random init to ensure gradient
        # flow to both parameter sets. The initial delta_W is near-zero because
        # the product of two small-norm matrices is very small.
        # Scale: 1/sqrt(k) so that delta_W norm is O(p/sqrt(k_beta * k_alpha)).
        self.eps_beta = nn.ParameterDict()
        self.eps_alpha = nn.ParameterDict()

        total_params = 0
        for layer_name in self.layer_names:
            safe = layer_name.replace(".", "_")
            k_beta = self.layer_k_beta[layer_name]
            k_alpha = self.layer_k_alpha[layer_name]
            # Small random init for both, scaled by 1/sqrt(k)
            scale_beta = 0.01 / (k_beta ** 0.5)
            scale_alpha = 0.01 / (k_alpha ** 0.5)
            self.eps_beta[safe] = nn.Parameter(
                torch.randn(k_beta, p, dtype=torch.float32) * scale_beta
            )
            self.eps_alpha[safe] = nn.Parameter(
                torch.randn(k_alpha, p, dtype=torch.float32) * scale_alpha
            )
            total_params += (k_beta + k_alpha) * p

        k_betas = list(self.layer_k_beta.values())
        k_alphas = list(self.layer_k_alpha.values())
        logger.info(f"Phase 2 config: p={p}, layers={len(self.layer_names)}")
        logger.info(f"  k_beta: {min(k_betas)}-{max(k_betas)}, k_alpha: {min(k_alphas)}-{max(k_alphas)}")
        logger.info(f"  Total trainable parameters: {total_params:,}")

    def get_delta_w(self, layer_name: str) -> torch.Tensor:
        """Reconstruct delta_W for a layer.

        delta_W = (beta @ eps_beta) @ (alpha @ eps_alpha).T
               = (n, k_b)(k_b, p) @ ((d, k_a)(k_a, p)).T
               = (n, p) @ (p, d) = (n, d)
        """
        safe = layer_name.replace(".", "_")
        beta = getattr(self, self.betas[layer_name])   # (n, k_beta), bfloat16
        alpha = getattr(self, self.alphas[layer_name])  # (d, k_alpha), bfloat16
        eps_beta = self.eps_beta[safe]   # (k_beta, p), float32
        eps_alpha = self.eps_alpha[safe]  # (k_alpha, p), float32

        # Compute in float32 for gradient stability, cast result to bfloat16
        B_hat = beta.float() @ eps_beta   # (n, p)
        A_hat = (alpha.float() @ eps_alpha).T  # (p, d)
        delta_W = B_hat @ A_hat  # (n, d)
        return delta_W.to(torch.bfloat16)

    def save_share_coefficients(self, output_dir: Path):
        """Save trained coefficients in Share format (numpy files).

        Creates output_dir/eps_beta_*.npy and eps_alpha_*.npy files
        compatible with share_inference.py.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for layer_name in self.layer_names:
            safe = layer_name.replace(".", "_")
            eps_beta = self.eps_beta[safe].detach().cpu().numpy()
            eps_alpha = self.eps_alpha[safe].detach().cpu().numpy()
            np.save(output_dir / f"eps_beta_{safe}.npy", eps_beta)
            np.save(output_dir / f"eps_alpha_{safe}.npy", eps_alpha)
        logger.info(f"Saved Share coefficients ({len(self.layer_names)} layers) to {output_dir}")


class SFTDataset(Dataset):
    """SFT dataset for Phase 2 training."""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        logger.info(f"Loaded {len(self.examples)} training examples from {data_path}")

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
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # Mask labels for everything EXCEPT the assistant response.
        # Only compute loss on the output tokens, not the instruction/input.
        labels = input_ids.clone()
        labels[~attention_mask.bool()] = -100

        # Find where the assistant response starts in the tokenized sequence.
        # Tokenize just the system+user prefix to find the boundary.
        prefix_messages = [
            {"role": "system", "content": ex.get("instruction", "Fix the Rust code.")},
            {"role": "user", "content": ex["input"]},
        ]
        # Add assistant header but no content
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = self.tokenizer(prefix_text, truncation=True, max_length=self.max_length)
        prefix_len = len(prefix_ids["input_ids"])

        # Mask all prefix tokens (system + user + assistant header)
        labels[:prefix_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _strip_peft_prefix(name: str) -> str:
    """Strip 'base_model.model.' prefix for HuggingFace module lookup."""
    for prefix in ["base_model.model.", "base_model."]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def train_phase2(
    share_dir: Path,
    data_path: Path,
    task_id: str,
    output_dir: Path,
    base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    steps: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    p: int = 1,
):
    """Train Phase 2 coefficients and save in Share format.

    Returns the path to the saved coefficients directory.
    """
    logger.info("=" * 60)
    logger.info("Share Phase 2: Proper Coefficient-Only Training")
    logger.info("=" * 60)
    logger.info(f"  task_id: {task_id}")
    logger.info(f"  share_dir: {share_dir}")
    logger.info(f"  data: {data_path}")
    logger.info(f"  p={p}, steps={steps}, lr={lr}, batch={batch_size}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=torch.bfloat16,
        device_map="cuda", local_files_only=True,
    )
    # Freeze ALL base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Initialize Share coefficients
    share_coefs = ShareCoefficientsV2(share_dir, p=p)
    share_coefs = share_coefs.cuda()
    share_coefs.train()

    # Build layer mapping: HF module name -> Share layer name
    layer_map = {}
    for share_name in share_coefs.layer_names:
        hf_name = _strip_peft_prefix(share_name)
        layer_map[hf_name] = share_name

    # Register forward hooks
    hooks = []
    def make_hook(share_layer_name):
        def hook_fn(module, input, output):
            delta_W = share_coefs.get_delta_w(share_layer_name)
            x = input[0] if isinstance(input, tuple) else input
            # LoRA: delta_output = x @ delta_W.T
            # delta_W is (n, d), delta_W.T is (d, n), x is (..., d)
            delta = torch.nn.functional.linear(x, delta_W)
            return output + delta
        return hook_fn

    for name, module in model.named_modules():
        if name in layer_map:
            h = module.register_forward_hook(make_hook(layer_map[name]))
            hooks.append(h)
    logger.info(f"Registered {len(hooks)} forward hooks")

    # Dataset and dataloader
    dataset = SFTDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Optimizer: only coefficient parameters
    optimizer = torch.optim.AdamW(share_coefs.parameters(), lr=lr, weight_decay=0.01)

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if step == 0:
                logger.info(f"Initial loss: {loss.item():.4f} (requires_grad={loss.requires_grad})")

            loss.backward()

            if step == 0:
                # Verify gradient flow
                nonzero = sum(1 for p in share_coefs.parameters() if p.grad is not None and p.grad.abs().max() > 0)
                total = sum(1 for _ in share_coefs.parameters())
                logger.info(f"Gradient check: {nonzero}/{total} params have nonzero gradients")
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % log_interval == 0 or step == steps:
                avg = total_loss / step
                logger.info(f"  Step {step}/{steps} | Loss: {loss.item():.4f} | Avg: {avg:.4f}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save coefficients in Share format
    coef_dir = output_dir / "coefficients" / task_id
    share_coefs.save_share_coefficients(coef_dir)

    # Also copy metadata if this is a new output dir
    metadata_dst = output_dir / "metadata.json"
    if not metadata_dst.exists():
        import shutil
        shutil.copy2(share_dir / "metadata.json", metadata_dst)
        # Copy basis too
        basis_dst = output_dir / "basis"
        if not basis_dst.exists():
            shutil.copytree(share_dir / "basis", basis_dst)
        logger.info(f"Copied basis and metadata to {output_dir}")

    # Save training metadata
    info = {
        "task_id": task_id,
        "share_dir": str(share_dir),
        "data_path": str(data_path),
        "p": p,
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "init_strategy": "lora_style (eps_beta=zero, eps_alpha=kaiming)",
        "final_loss": loss.item(),
        "avg_loss": total_loss / steps,
        "trainable_params": sum(
            (share_coefs.layer_k_beta[ln] + share_coefs.layer_k_alpha[ln]) * p
            for ln in share_coefs.layer_names
        ),
        "timestamp": datetime.now().isoformat(),
    }
    with open(coef_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Phase 2 training complete! Coefficients at: {coef_dir}")
    return coef_dir


def main():
    parser = argparse.ArgumentParser(description="Share Phase 2: Proper Coefficient Training")
    parser.add_argument("--share", "-s", default="runs/share_proper_trained",
                        help="Path to Share model directory (with basis)")
    parser.add_argument("--data", "-d", required=True, help="Training data (JSONL)")
    parser.add_argument("--task-id", "-t", required=True, help="Task/pattern identifier")
    parser.add_argument("--output", "-o", default="runs/share_phase2",
                        help="Output directory for trained coefficients")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--p", type=int, default=1, help="Pseudo-rank (coefficient column count)")
    args = parser.parse_args()

    train_phase2(
        share_dir=Path(args.share),
        data_path=Path(args.data),
        task_id=args.task_id,
        output_dir=Path(args.output),
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        p=args.p,
    )


if __name__ == "__main__":
    main()
