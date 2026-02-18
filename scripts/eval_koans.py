#!/usr/bin/env python3
"""
Python Eval Harness for Sleepy-Coder Koans.

Mirrors the Rust Sandbox + EvalHarness but runs inference through
the Python ShareInferenceEngine for proper coefficient routing.

Usage:
    from eval_koans import KoanEvaluator, RustSandbox
    evaluator = KoanEvaluator(engine)
    results = evaluator.run_eval(koans, strategy="routed")
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from share_inference import ShareInferenceEngine, detect_pattern


# Frozen eval set: bc_001-010, tb_001-010, rh_001-010
FROZEN_EVAL_PREFIXES = ("bc_", "tb_", "rh_")
FROZEN_EVAL_IDS = {f"{p}{i:03d}" for p in ("bc_", "tb_", "rh_") for i in range(1, 11)}

# Named coefficient sets (v4: dual-random init, p=4, 100 steps, both params trained)
NAMED_COEFFICIENTS = [
    "mut_borrow_conflict_v4",
    "double_mut_borrow_v4",
    "return_local_ref_v4",
    "missing_clone_v4",
    "missing_hash_v4",
    "missing_ord_v4",
    "option_ok_or_v4",
    "result_map_err_v4",
]

# Mapping from pattern name (as returned by detect_pattern) to coefficient id
PATTERN_TO_COEF = {
    "mut_borrow_conflict": "mut_borrow_conflict_v4",
    "double_mut_borrow": "double_mut_borrow_v4",
    "return_local_ref": "return_local_ref_v4",
    "missing_clone": "missing_clone_v4",
    "missing_hash": "missing_hash_v4",
    "missing_ord": "missing_ord_v4",
    "option_ok_or": "option_ok_or_v4",
    "result_map_err": "result_map_err_v4",
}


@dataclass
class KoanResult:
    """Result of evaluating a single koan."""
    task_id: str
    family: str
    passed: bool
    error_message: str = ""
    generated_code: str = ""
    pattern_used: Optional[str] = None
    strategy: str = ""
    elapsed_s: float = 0.0


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    strategy: str
    total: int = 0
    passed: int = 0
    pass_rate: float = 0.0
    per_family: Dict[str, Dict[str, int]] = field(default_factory=dict)
    koan_results: List[KoanResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Strategy: {self.strategy}",
            f"Pass rate: {self.passed}/{self.total} = {self.pass_rate:.1%}",
            "",
            "Per-family breakdown:",
        ]
        for family, counts in sorted(self.per_family.items()):
            p = counts["passed"]
            t = counts["total"]
            rate = p / t if t > 0 else 0
            lines.append(f"  {family}: {p}/{t} = {rate:.1%}")

        lines.append("")
        lines.append("Per-koan results:")
        for r in self.koan_results:
            status = "PASS" if r.passed else "FAIL"
            pat = f" [{r.pattern_used}]" if r.pattern_used else ""
            lines.append(f"  {r.task_id} ({r.family}): {status}{pat}")
        return "\n".join(lines)


class RustSandbox:
    """Compile Rust code in a temporary Cargo project."""

    def __init__(self, task_id: str = "sandbox"):
        self.task_id = task_id
        self._tmpdir = tempfile.mkdtemp(prefix=f"sleepy_{task_id}_")
        self._src_dir = Path(self._tmpdir) / "src"
        self._src_dir.mkdir()

        # Write Cargo.toml
        cargo_toml = f"""[package]
name = "sandbox_{task_id}"
version = "0.1.0"
edition = "2021"
"""
        (Path(self._tmpdir) / "Cargo.toml").write_text(cargo_toml)

    def write_code(self, code: str):
        """Write code to src/main.rs."""
        (self._src_dir / "main.rs").write_text(code)

    def check(self, timeout: int = 30) -> tuple[bool, str]:
        """Run cargo check and return (success, stderr)."""
        try:
            result = subprocess.run(
                ["cargo", "check"],
                cwd=self._tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stderr
        except subprocess.TimeoutExpired:
            return False, "cargo check timed out"

    def cleanup(self):
        """Remove temporary directory."""
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class KoanEvaluator:
    """Evaluate koans using a ShareInferenceEngine."""

    def __init__(self, engine: ShareInferenceEngine):
        self.engine = engine

    def _build_prompt(self, buggy_code: str, error_message: str) -> str:
        """Build a baseline prompt (no hints, no few-shot) for fair comparison."""
        return f"""Fix the following Rust code that has a compilation error.

## Buggy Code:
```rust
{buggy_code}
```

## Compiler Error:
{error_message}

## Instructions:
- Return ONLY the fixed Rust code
- Do not include any explanation
- Do not include markdown code fences
- The code should compile without errors

## Fixed Code:
"""

    def evaluate_koan(
        self, koan: dict, strategy: str = "baseline"
    ) -> KoanResult:
        """Evaluate a single koan with the given strategy.

        Strategies:
          - "baseline": No LoRA, pure base model
          - "averaged": Average all named coefficient sets
          - "routed": Detect error pattern, use matching coefficients
        """
        task_id = koan["id"]
        family = koan["family"]
        buggy_code = koan["buggy_code"]
        start = time.time()

        # Step 1: Compile buggy code to get error message
        sandbox = RustSandbox(task_id)
        try:
            sandbox.write_code(buggy_code)
            _, error_message = sandbox.check()

            # Step 2: Select coefficients based on strategy
            pattern_used = None
            if strategy == "baseline":
                self.engine.restore_weights()
            elif strategy == "averaged":
                self.engine.apply_averaged_coefficients(NAMED_COEFFICIENTS)
                pattern_used = "averaged"
            elif strategy == "routed":
                pattern = detect_pattern(error_message)
                coef_id = PATTERN_TO_COEF.get(pattern) if pattern else None
                if coef_id:
                    self.engine.apply_coefficients(coef_id)
                    pattern_used = pattern
                else:
                    # No matching pattern: fall back to base model (no-harm principle)
                    self.engine.restore_weights()
                    pattern_used = None

            # Step 3: Generate fix
            prompt = self._build_prompt(buggy_code, error_message)
            raw_response = self.engine.generate(prompt, max_new_tokens=2048)
            fixed_code = self.engine.extract_code(raw_response)

            # Step 4: Check if fix compiles
            sandbox.write_code(fixed_code)
            passed, check_stderr = sandbox.check()

            elapsed = time.time() - start
            return KoanResult(
                task_id=task_id,
                family=family,
                passed=passed,
                error_message=check_stderr if not passed else "",
                generated_code=fixed_code,
                pattern_used=pattern_used,
                strategy=strategy,
                elapsed_s=elapsed,
            )
        finally:
            sandbox.cleanup()

    def run_eval(
        self, koans: List[dict], strategy: str = "baseline"
    ) -> EvalResults:
        """Evaluate all koans and compute aggregate metrics."""
        results = EvalResults(strategy=strategy)
        results.total = len(koans)

        for i, koan in enumerate(koans):
            task_id = koan["id"]
            print(f"  [{i+1}/{len(koans)}] {task_id} ({strategy})...", end=" ", flush=True)

            result = self.evaluate_koan(koan, strategy)
            results.koan_results.append(result)

            if result.passed:
                results.passed += 1
                print(f"PASS ({result.elapsed_s:.1f}s)")
            else:
                pat = f" [{result.pattern_used}]" if result.pattern_used else ""
                print(f"FAIL{pat} ({result.elapsed_s:.1f}s)")

            # Per-family tracking
            fam = result.family
            if fam not in results.per_family:
                results.per_family[fam] = {"total": 0, "passed": 0}
            results.per_family[fam]["total"] += 1
            if result.passed:
                results.per_family[fam]["passed"] += 1

        results.pass_rate = results.passed / results.total if results.total > 0 else 0.0

        # Restore to base weights after eval
        self.engine.restore_weights()

        return results


def load_frozen_koans(tasks_path: str | Path = "data/export/tasks.json") -> List[dict]:
    """Load and filter to frozen eval set (bc_001-010, tb_001-010, rh_001-010)."""
    with open(tasks_path) as f:
        all_tasks = json.load(f)
    frozen = [t for t in all_tasks if t["id"] in FROZEN_EVAL_IDS]
    frozen.sort(key=lambda t: t["id"])
    return frozen


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test-sandbox":
        # Quick sandbox test
        sb = RustSandbox("test")
        sb.write_code('fn main() { println!("hello"); }')
        ok, stderr = sb.check()
        print(f"Compiles: {ok}")
        if not ok:
            print(f"Error: {stderr}")
        sb.cleanup()

        sb2 = RustSandbox("test2")
        sb2.write_code("fn main() { let s = String::new(); let t = s; println!(\"{}\", s); }")
        ok2, stderr2 = sb2.check()
        print(f"\nBuggy compiles: {ok2}")
        print(f"Error: {stderr2[:200]}")
        sb2.cleanup()
    else:
        print("Usage: python eval_koans.py test-sandbox")
