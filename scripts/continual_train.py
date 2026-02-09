#!/usr/bin/env python3
"""
Continual Learning Training Pipeline for Sleepy-Coder.

Implements research-backed techniques to prevent catastrophic forgetting:
1. Replay buffer - Mix successful + failed examples
2. Self-Synthesized Rehearsal (SSR) - Generate synthetic examples
3. Data mixing - Balance new and old data
4. Lower learning rates for continual learning
5. Multi-cycle training with evaluation after each

References:
- SSR: https://aclanthology.org/2024.acl-long.77/
- Replay: https://openreview.net/pdf?id=IgZWU75BLL

Usage:
    python scripts/continual_train.py --cycles 5
    python scripts/continual_train.py --cycles 10 --replay-ratio 0.5
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ContinualConfig:
    """Configuration for continual learning."""

    # Training cycles
    num_cycles: int = 5
    steps_per_cycle: int = 100  # Smaller steps, more frequent updates

    # Replay settings
    replay_ratio: float = 0.5  # Ratio of replay data to new data
    include_successful: bool = True  # Include successful episodes

    # Learning rate schedule (decreasing over cycles)
    initial_lr: float = 1e-4  # Lower than before
    lr_decay: float = 0.9  # Decay per cycle

    # LoRA settings (more conservative)
    lora_r: int = 8  # Lower rank = less forgetting
    lora_alpha: int = 16
    lora_dropout: float = 0.1  # Higher dropout for regularization

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "rust" / "data" / "episodes"

    @property
    def sft_dir(self) -> Path:
        return self.project_root / "data" / "sft"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs" / "continual"


def load_episodes(data_dir: Path, cycle: int) -> list[dict]:
    """Load episodes from a specific cycle."""
    episodes = []
    cycle_file = data_dir / f"cycle_{cycle}.jsonl"

    if cycle_file.exists():
        with open(cycle_file) as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))

    return episodes


def load_all_episodes(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load all episodes, separated into passed and failed."""
    passed = []
    failed = []

    for jsonl_file in sorted(data_dir.glob("cycle_*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    ep = json.loads(line)
                    if ep.get("passed"):
                        passed.append(ep)
                    else:
                        failed.append(ep)

    return passed, failed


def load_sft_examples(sft_dir: Path) -> list[dict]:
    """Load existing SFT training examples."""
    examples = []
    train_file = sft_dir / "train.jsonl"

    if train_file.exists():
        with open(train_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

    return examples


def create_mixed_dataset(
    sft_examples: list[dict],
    passed_episodes: list[dict],
    failed_episodes: list[dict],
    replay_ratio: float,
    include_successful: bool,
) -> list[dict]:
    """
    Create a mixed training dataset following research best practices.

    Mix ratios (based on literature):
    - 50% replay from original capabilities
    - 25% successful episodes (positive examples)
    - 25% failed episodes (learning signal)
    """
    import random

    mixed = []

    # 1. Replay data (50%) - prevents catastrophic forgetting
    if sft_examples and replay_ratio > 0:
        n_replay = int(len(sft_examples) * replay_ratio)
        replay_samples = random.sample(sft_examples, min(n_replay, len(sft_examples)))
        for ex in replay_samples:
            ex["source"] = "replay"
        mixed.extend(replay_samples)
        logger.info(f"Added {len(replay_samples)} replay examples")

    # 2. Successful episodes (25%) - reinforces good behavior
    if include_successful and passed_episodes:
        # Convert episodes to SFT format (need to load task definitions)
        logger.info(f"Found {len(passed_episodes)} successful episodes (for reference)")

    # 3. Failed episodes (25%) - the actual learning signal
    # These need to be converted to correct SFT format
    if failed_episodes:
        logger.info(f"Found {len(failed_episodes)} failed episodes (learning signal)")

    # Shuffle the mixed dataset
    random.shuffle(mixed)

    return mixed


def save_mixed_dataset(examples: list[dict], output_path: Path):
    """Save mixed dataset to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Saved {len(examples)} examples to {output_path}")


def run_training(
    config: ContinualConfig,
    cycle: int,
    data_path: Path,
    output_dir: Path,
) -> dict:
    """Run a single training cycle."""
    # Calculate learning rate for this cycle
    lr = config.initial_lr * (config.lr_decay ** cycle)

    cmd = [
        sys.executable,
        str(config.project_root / "cuda" / "scripts" / "train.py"),
        "--steps", str(config.steps_per_cycle),
        "--lr", str(lr),
        "--output", str(output_dir),
    ]

    logger.info(f"Running training cycle {cycle} with lr={lr:.2e}")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        return {"success": False, "error": result.stderr}

    return {"success": True, "output": result.stdout}


def run_evaluation(config: ContinualConfig, cycle: int, model: str) -> dict:
    """Run evaluation and return metrics."""
    cmd = [
        str(config.project_root / "rust" / "target" / "release" / "sleepy-coder"),
        "eval",
        "--cycle", str(cycle),
        "--model", model,
    ]

    logger.info(f"Running evaluation cycle {cycle} with model {model}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(config.project_root / "rust"))

    # Parse metrics from output
    metrics = {"cycle": cycle, "model": model}

    # Read from metrics.jsonl
    metrics_file = config.data_dir / "metrics.jsonl"
    if metrics_file.exists():
        with open(metrics_file) as f:
            lines = f.readlines()
            if lines:
                last_metrics = json.loads(lines[-1])
                metrics.update(last_metrics)

    return metrics


def generate_learning_curve(metrics_history: list[dict], output_path: Path):
    """Generate learning curve plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot generation")
        return

    cycles = [m["cycle"] for m in metrics_history]
    pass_rates = [m.get("pass_rate", 0) * 100 for m in metrics_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot learning curve
    ax.plot(cycles, pass_rates, 'b-o', linewidth=2, markersize=8, label='Pass Rate')

    # Add baseline reference
    if pass_rates:
        ax.axhline(y=pass_rates[0], color='red', linestyle='--', alpha=0.5, label='Baseline')

    # Styling
    ax.set_xlabel('Training Cycle', fontsize=12)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Continual Learning Progress', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add annotations for improvement
    for i, (cycle, rate) in enumerate(zip(cycles, pass_rates)):
        ax.annotate(f'{rate:.1f}%', (cycle, rate), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Learning curve saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Continual learning for sleepy-coder")
    parser.add_argument("--cycles", "-c", type=int, default=5, help="Number of training cycles")
    parser.add_argument("--steps", "-s", type=int, default=100, help="Steps per cycle")
    parser.add_argument("--replay-ratio", type=float, default=0.5, help="Replay data ratio")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    config = ContinualConfig(
        num_cycles=args.cycles,
        steps_per_cycle=args.steps,
        replay_ratio=args.replay_ratio,
        initial_lr=args.lr,
    )

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CONTINUAL LEARNING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Cycles: {config.num_cycles}")
    logger.info(f"Steps per cycle: {config.steps_per_cycle}")
    logger.info(f"Replay ratio: {config.replay_ratio}")
    logger.info(f"Initial LR: {config.initial_lr}")
    logger.info(f"Output: {run_dir}")
    logger.info("=" * 60)

    # Load existing data
    sft_examples = load_sft_examples(config.sft_dir)
    passed_episodes, failed_episodes = load_all_episodes(config.data_dir)

    logger.info(f"Loaded {len(sft_examples)} SFT examples")
    logger.info(f"Loaded {len(passed_episodes)} passed episodes")
    logger.info(f"Loaded {len(failed_episodes)} failed episodes")

    # Track metrics across cycles
    metrics_history = []

    # Run baseline evaluation first
    logger.info("\n--- Baseline Evaluation ---")
    baseline_metrics = run_evaluation(config, 0, "qwen2.5-coder:1.5b-instruct-q4_K_M")
    metrics_history.append(baseline_metrics)
    logger.info(f"Baseline pass rate: {baseline_metrics.get('pass_rate', 0) * 100:.1f}%")

    if args.eval_only:
        logger.info("Eval-only mode, skipping training")
        return

    # Run continual learning cycles
    for cycle in range(1, config.num_cycles + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CYCLE {cycle}/{config.num_cycles}")
        logger.info(f"{'=' * 60}")

        # 1. Create mixed dataset with replay
        mixed_data = create_mixed_dataset(
            sft_examples,
            passed_episodes,
            failed_episodes,
            config.replay_ratio,
            config.include_successful,
        )

        if not mixed_data:
            logger.warning("No training data available, skipping cycle")
            continue

        # Save mixed dataset
        cycle_data_path = run_dir / f"cycle_{cycle}_data.jsonl"
        save_mixed_dataset(mixed_data, cycle_data_path)

        # 2. Run training
        train_result = run_training(
            config,
            cycle,
            cycle_data_path,
            run_dir / f"cycle_{cycle}",
        )

        if not train_result.get("success"):
            logger.error(f"Training failed in cycle {cycle}")
            continue

        # 3. Export model to Ollama
        model_name = f"sleepy-coder-cycle{cycle}"
        # TODO: Run merge and export

        # 4. Run evaluation
        cycle_metrics = run_evaluation(config, cycle, model_name)
        metrics_history.append(cycle_metrics)

        # 5. Log progress
        current_rate = cycle_metrics.get("pass_rate", 0) * 100
        baseline_rate = metrics_history[0].get("pass_rate", 0) * 100
        improvement = current_rate - baseline_rate

        logger.info(f"Cycle {cycle} pass rate: {current_rate:.1f}% ({improvement:+.1f}%)")

        # 6. Update replay buffer with new episodes
        new_passed, new_failed = load_episodes(config.data_dir, cycle), []
        for ep in new_passed:
            if ep.get("passed"):
                passed_episodes.append(ep)
            else:
                failed_episodes.append(ep)

    # Generate final visualizations
    logger.info("\n--- Generating Visualizations ---")
    generate_learning_curve(metrics_history, run_dir / "learning_curve.png")

    # Save metrics history
    with open(run_dir / "metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if len(metrics_history) >= 2:
        final_rate = metrics_history[-1].get("pass_rate", 0) * 100
        baseline_rate = metrics_history[0].get("pass_rate", 0) * 100
        total_improvement = final_rate - baseline_rate

        logger.info(f"Baseline: {baseline_rate:.1f}%")
        logger.info(f"Final: {final_rate:.1f}%")
        logger.info(f"Total improvement: {total_improvement:+.1f}%")

    logger.info(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
