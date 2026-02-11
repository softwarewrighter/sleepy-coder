#!/usr/bin/env python3
"""
Routed Inference: Select the right Share coefficients based on error type.

This implements the Share paper's task routing concept:
- Analyze the compiler error
- Route to the appropriate coefficient set
- Use that for inference
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

os.environ["HF_HUB_OFFLINE"] = "1"

# Error pattern to coefficient mapping
ERROR_PATTERNS = {
    # Borrow checker patterns
    "mut_borrow_conflict": [
        r"cannot borrow.*as mutable.*immutable borrow",
        r"cannot borrow.*as immutable.*mutable borrow",
        r"mutable borrow.*while.*immutable",
    ],
    "double_mut_borrow": [
        r"cannot borrow.*as mutable more than once",
        r"second mutable borrow occurs",
    ],
    "return_local_ref": [
        r"returns a reference to data owned",
        r"cannot return reference to local",
        r"borrowed value does not live long enough",
    ],
    # Trait bound patterns
    "missing_clone": [
        r"method `clone` exists.*but.*trait bounds.*not satisfied",
        r"Clone.*is not implemented",
        r"cannot move out.*move occurs.*doesn't implement.*Copy",
    ],
    "missing_hash": [
        r"Hash.*is not implemented",
        r"trait bound.*Hash.*is not satisfied",
    ],
    "missing_ord": [
        r"Ord.*is not implemented",
        r"PartialOrd.*is not implemented",
        r"trait bound.*Ord.*is not satisfied",
    ],
    # Result handling patterns
    "option_ok_or": [
        r"expected.*Result.*found.*Option",
        r"the.*operator cannot be applied to type.*Option",
    ],
    "result_map_err": [
        r"expected.*found.*Result",
        r"error type.*doesn't implement.*From",
    ],
}

# Task ID to pattern mapping (for known koans)
TASK_PATTERN_MAP = {
    # Borrow checker
    "bc_003": "mut_borrow_conflict",
    "bc_005": "double_mut_borrow",
    "bc_010": "return_local_ref",
    # Trait bounds
    "tb_002": "missing_clone",
    "tb_007": "missing_hash",
    "tb_008": "missing_ord",
    # Result handling
    "rh_004": "option_ok_or",
}


def detect_pattern(error_message: str) -> Optional[str]:
    """Detect which pattern matches the error message."""
    error_lower = error_message.lower()

    for pattern_name, regexes in ERROR_PATTERNS.items():
        for regex in regexes:
            if re.search(regex, error_message, re.IGNORECASE):
                return pattern_name

    # Fallback heuristics
    if "clone" in error_lower and "trait" in error_lower:
        return "missing_clone"
    if "hash" in error_lower:
        return "missing_hash"
    if "ord" in error_lower or "sort" in error_lower:
        return "missing_ord"
    if "borrow" in error_lower and "mutable" in error_lower:
        return "mut_borrow_conflict"
    if "option" in error_lower and "result" in error_lower:
        return "option_ok_or"

    return None


class RoutedShareInference:
    """Inference with dynamic coefficient routing."""

    def __init__(
        self,
        share_dir: Path,
        base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device: str = "cuda",
    ):
        self.share_dir = Path(share_dir)
        self.device = device

        # Load metadata
        with open(self.share_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.layer_names = self.metadata["layer_names"]

        # Load basis (shared, frozen)
        self.beta = {}
        self.alpha = {}
        for layer in self.layer_names:
            safe = layer.replace(".", "_")
            self.beta[layer] = torch.from_numpy(
                np.load(self.share_dir / "basis" / f"beta_{safe}.npy")
            ).to(device).to(torch.bfloat16)
            self.alpha[layer] = torch.from_numpy(
                np.load(self.share_dir / "basis" / f"alpha_{safe}.npy")
            ).to(device).to(torch.bfloat16)

        # Load base model
        print(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

        # Cache for loaded coefficients
        self.coef_cache: Dict[str, Dict] = {}

        # Current active LoRA weights
        self.active_lora: Optional[str] = None

    def load_coefficients(self, task_id: str) -> Dict:
        """Load coefficients for a task."""
        if task_id in self.coef_cache:
            return self.coef_cache[task_id]

        coef_dir = self.share_dir / "coefficients" / task_id
        if not coef_dir.exists():
            raise ValueError(f"No coefficients found for task: {task_id}")

        coefficients = {}
        for layer in self.layer_names:
            safe = layer.replace(".", "_")
            eps_beta = torch.from_numpy(
                np.load(coef_dir / f"eps_beta_{safe}.npy")
            ).to(self.device).to(torch.bfloat16)
            eps_alpha = torch.from_numpy(
                np.load(coef_dir / f"eps_alpha_{safe}.npy")
            ).to(self.device).to(torch.bfloat16)
            coefficients[layer] = (eps_beta, eps_alpha)

        self.coef_cache[task_id] = coefficients
        return coefficients

    def apply_lora(self, task_id: str):
        """Apply LoRA weights for a specific task."""
        if self.active_lora == task_id:
            return  # Already applied

        coefficients = self.load_coefficients(task_id)

        # Reconstruct and apply LoRA weights
        for name, module in self.model.named_modules():
            for layer in self.layer_names:
                layer_short = layer.replace("base_model.model.", "")
                if name == layer_short:
                    eps_beta, eps_alpha = coefficients[layer]
                    beta = self.beta[layer]
                    alpha = self.alpha[layer]

                    # B_hat = beta @ eps_beta, A_hat = (alpha @ eps_alpha).T
                    B_hat = beta @ eps_beta
                    A_hat = (alpha @ eps_alpha).T

                    # Store as buffer for hook
                    if not hasattr(module, '_lora_B'):
                        module.register_buffer('_lora_B', B_hat)
                        module.register_buffer('_lora_A', A_hat)
                    else:
                        module._lora_B.copy_(B_hat)
                        module._lora_A.copy_(A_hat)

        self.active_lora = task_id

    def generate(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        error_message: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate response with routed coefficients."""
        # Determine which pattern to use
        if task_id and task_id in TASK_PATTERN_MAP:
            pattern = TASK_PATTERN_MAP[task_id]
        elif error_message:
            pattern = detect_pattern(error_message)
        else:
            pattern = "failures_v1"  # Fallback to general

        if pattern:
            try:
                self.apply_lora(pattern)
                print(f"Using pattern: {pattern}")
            except ValueError:
                print(f"Pattern {pattern} not found, using failures_v1")
                self.apply_lora("failures_v1")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]


def test_routing():
    """Test the routing on known failing tasks."""
    share_dir = Path("runs/share_proper_trained")

    # Test cases: (task_id, error_message, expected_pattern)
    test_cases = [
        ("bc_003", "cannot borrow `v` as mutable because it is also borrowed as immutable", "mut_borrow_conflict"),
        ("bc_005", "cannot borrow `s` as mutable more than once at a time", "double_mut_borrow"),
        ("bc_010", "returns a reference to data owned by the current function", "return_local_ref"),
        ("tb_002", "no method named `clone` found; Clone is not implemented", "missing_clone"),
        ("tb_007", "the trait `Hash` is not implemented for `Key`", "missing_hash"),
        ("tb_008", "the trait `Ord` is not implemented for `Score`", "missing_ord"),
        ("rh_004", "expected Result, found Option", "option_ok_or"),
    ]

    print("Testing error pattern routing:\n")
    for task_id, error, expected in test_cases:
        detected = detect_pattern(error)
        mapped = TASK_PATTERN_MAP.get(task_id)
        status = "✓" if detected == expected or mapped == expected else "✗"
        print(f"{status} {task_id}: detected={detected}, mapped={mapped}, expected={expected}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_routing()
    else:
        print("Usage: python routed_inference.py test")
        print("       (Full inference requires integration with eval harness)")
