#!/usr/bin/env python3
"""
Share Inference Engine with Direct Weight Modification.

Fixes the bug in RoutedShareInference where LoRA weights were stored as
buffers but never intercepted the forward pass. This version modifies
model weights directly, so model.generate() uses the modified weights
automatically.

Usage:
    from share_inference import ShareInferenceEngine
    engine = ShareInferenceEngine("runs/share_proper_trained")
    engine.apply_coefficients("mut_borrow_conflict")
    output = engine.generate("Fix this Rust code...")
    engine.restore_weights()
"""

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"

# Reuse pattern detection from routed_inference.py
ERROR_PATTERNS = {
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
    "option_ok_or": [
        r"expected.*Result.*found.*Option",
        r"the.*operator cannot be applied to type.*Option",
    ],
    "result_map_err": [
        r"expected.*found.*Result",
        r"error type.*doesn't implement.*From",
    ],
}


def detect_pattern(error_message: str) -> Optional[str]:
    """Detect which Share coefficient pattern matches the error message."""
    for pattern_name, regexes in ERROR_PATTERNS.items():
        for regex in regexes:
            if re.search(regex, error_message, re.IGNORECASE):
                return pattern_name

    # Fallback heuristics
    error_lower = error_message.lower()
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


def _strip_peft_prefix(layer_name: str) -> str:
    """Strip 'base_model.model.' prefix to get HuggingFace module path."""
    return layer_name.replace("base_model.model.", "")


def _layer_to_filename(layer_name: str) -> str:
    """Convert dotted layer name to underscore filename component."""
    return layer_name.replace(".", "_")


class ShareInferenceEngine:
    """Inference engine with direct weight modification (fixes the hook bug)."""

    def __init__(
        self,
        share_dir: str | Path,
        base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device: str = "cuda",
    ):
        self.share_dir = Path(share_dir)
        self.device = device

        # Load metadata
        with open(self.share_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.layer_names = self.metadata["layer_names"]
        self.p = self.metadata.get("p", 1)

        # Load shared basis vectors (frozen)
        print("Loading Share basis...")
        self.beta: Dict[str, torch.Tensor] = {}
        self.alpha: Dict[str, torch.Tensor] = {}
        for layer in self.layer_names:
            safe = _layer_to_filename(layer)
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

        # Save original weights for restoration
        print("Saving original weights...")
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._module_map: Dict[str, torch.nn.Module] = {}
        for layer in self.layer_names:
            hf_name = _strip_peft_prefix(layer)
            module = self._get_module(hf_name)
            if module is not None:
                self._original_weights[layer] = module.weight.data.clone()
                self._module_map[layer] = module

        # Coefficient cache
        self._coef_cache: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._active_coef: Optional[str] = None

        print(f"Ready. {len(self._module_map)} layers mapped, "
              f"{len(self._list_coefficients())} coefficient sets available.")

    def _get_module(self, dotted_name: str) -> Optional[torch.nn.Module]:
        """Get a module by its dotted name path."""
        parts = dotted_name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _list_coefficients(self) -> List[str]:
        """List available coefficient set names."""
        coef_dir = self.share_dir / "coefficients"
        if not coef_dir.exists():
            return []
        return sorted(d.name for d in coef_dir.iterdir() if d.is_dir())

    def _load_coefficients(self, coef_id: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Load coefficient tensors for a given task/pattern."""
        if coef_id in self._coef_cache:
            return self._coef_cache[coef_id]

        coef_dir = self.share_dir / "coefficients" / coef_id
        if not coef_dir.exists():
            raise ValueError(f"No coefficients found: {coef_id}")

        coefficients = {}
        for layer in self.layer_names:
            safe = _layer_to_filename(layer)
            eps_beta = torch.from_numpy(
                np.load(coef_dir / f"eps_beta_{safe}.npy")
            ).to(self.device).to(torch.bfloat16)
            eps_alpha = torch.from_numpy(
                np.load(coef_dir / f"eps_alpha_{safe}.npy")
            ).to(self.device).to(torch.bfloat16)
            coefficients[layer] = (eps_beta, eps_alpha)

        self._coef_cache[coef_id] = coefficients
        return coefficients

    def _reconstruct_delta(
        self, layer: str, eps_beta: torch.Tensor, eps_alpha: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct delta_W = (beta @ eps_beta) @ (alpha @ eps_alpha).T"""
        beta = self.beta[layer]
        alpha = self.alpha[layer]
        B_hat = beta @ eps_beta      # (n, p)
        A_hat = (alpha @ eps_alpha).T  # (p, d)
        return B_hat @ A_hat           # (n, d)

    def restore_weights(self):
        """Restore all module weights to their original values."""
        for layer, module in self._module_map.items():
            module.weight.data.copy_(self._original_weights[layer])
        self._active_coef = None

    def apply_coefficients(self, coef_id: str):
        """Apply a single coefficient set by modifying weights directly."""
        if self._active_coef == coef_id:
            return
        # Always restore first to avoid stacking deltas
        self.restore_weights()

        coefficients = self._load_coefficients(coef_id)
        for layer, module in self._module_map.items():
            eps_beta, eps_alpha = coefficients[layer]
            delta_W = self._reconstruct_delta(layer, eps_beta, eps_alpha)
            module.weight.data.add_(delta_W)

        self._active_coef = coef_id

    def apply_averaged_coefficients(self, coef_ids: List[str]):
        """Average multiple coefficient sets and apply as a single delta."""
        self.restore_weights()

        # Load all coefficient sets
        all_coefs = [self._load_coefficients(cid) for cid in coef_ids]

        for layer, module in self._module_map.items():
            # Average eps_beta and eps_alpha across coefficient sets
            avg_eps_beta = torch.stack([c[layer][0] for c in all_coefs]).mean(dim=0)
            avg_eps_alpha = torch.stack([c[layer][1] for c in all_coefs]).mean(dim=0)
            delta_W = self._reconstruct_delta(layer, avg_eps_beta, avg_eps_alpha)
            module.weight.data.add_(delta_W)

        self._active_coef = f"averaged({','.join(coef_ids[:3])}...)"

    def generate(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """Generate text using the model with currently applied weights."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the input prompt from the response
        return response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]

    def extract_code(self, response: str) -> str:
        """Extract Rust code from LLM response, stripping markdown fences."""
        trimmed = response.strip()

        # Try ```rust ... ``` first
        if trimmed.startswith("```rust"):
            end = trimmed.rfind("```")
            start = len("```rust")
            if end > start:
                return trimmed[start:end].strip()

        # Try ``` ... ```
        if trimmed.startswith("```"):
            end = trimmed.rfind("```")
            start = 3
            if end > start:
                return trimmed[start:end].strip()

        return trimmed


def sanity_check():
    """Quick sanity check: apply coefficients, generate, restore, generate again."""
    share_dir = Path("runs/share_proper_trained")
    if not share_dir.exists():
        print(f"Share dir not found: {share_dir}")
        return

    engine = ShareInferenceEngine(share_dir)

    prompt = "Fix the following Rust code:\nfn main() { let mut v = vec![1,2,3]; let r = &v[0]; v.push(4); println!(\"{}\", r); }\n\nFixed code:\n"

    # Generate with base model
    print("\n--- Base model ---")
    engine.restore_weights()
    base_output = engine.generate(prompt, max_new_tokens=256)
    print(base_output[:200])

    # Generate with mut_borrow_conflict coefficients
    print("\n--- With mut_borrow_conflict coefficients ---")
    engine.apply_coefficients("mut_borrow_conflict")
    routed_output = engine.generate(prompt, max_new_tokens=256)
    print(routed_output[:200])

    # Restore and verify
    print("\n--- After restore ---")
    engine.restore_weights()
    restored_output = engine.generate(prompt, max_new_tokens=256)
    print(restored_output[:200])

    print(f"\nBase == Restored: {base_output == restored_output}")
    print(f"Base == Routed: {base_output == routed_output}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        sanity_check()
    else:
        print("Usage: python share_inference.py sanity")
