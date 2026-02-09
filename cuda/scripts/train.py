#!/usr/bin/env python3
"""
CUDA-optimized LoRA training for sleepy-coder.

Usage:
    python train.py --steps 500
    python train.py --steps 1000 --lr 1e-4
    python train.py --config ../configs/full_train.yaml
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force offline mode - no HuggingFace network requests
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration.

    NOTE: Conservative defaults to prevent catastrophic forgetting.
    See docs/changes.md for rationale.
    """

    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # LoRA - Conservative settings to prevent forgetting
    lora_r: int = 8  # Lower rank = less forgetting (was 16)
    lora_alpha: int = 16  # Alpha = 2 * r
    lora_dropout: float = 0.1  # Higher dropout for regularization (was 0.05)
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training - Conservative to preserve base model knowledge
    max_steps: int = 100  # Shorter cycles, more frequent eval (was 500)
    batch_size: int = 4  # Works well on 16GB+ GPUs
    gradient_accumulation_steps: int = 1  # Increase if OOM
    learning_rate: float = 1e-4  # Lower LR = less forgetting (was 2e-4)
    warmup_steps: int = 10  # Reduced warmup for short training
    max_seq_length: int = 2048  # Full context for code
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True  # Saves memory

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Checkpointing
    save_steps: int = 50
    logging_steps: int = 10

    # Paths (relative to cuda/ dir)
    data_path: str = "../data/sft/train.jsonl"  # Or use --data for mixed.jsonl
    output_dir: str = "../runs/adapters"

    # Misc
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "sleepy-coder"


def load_config(config_path: Optional[str] = None) -> TrainConfig:
    """Load config from YAML file or use defaults."""
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return TrainConfig(**config_dict)
    return TrainConfig()


def load_sft_data(data_path: Path) -> list[dict]:
    """Load SFT training data from JSONL file."""
    examples = []

    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        return examples

    logger.info(f"Loading {data_path}")
    with open(data_path) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                examples.append(example)

    logger.info(f"Loaded {len(examples)} training examples")
    return examples


def prepare_sft_dataset(examples: list[dict], tokenizer) -> Dataset:
    """Convert SFT examples to chat format for training."""

    def format_example(example: dict) -> str:
        """Format a single example using chat template."""
        # Build conversation from instruction/input/output format
        messages = []

        # System message from instruction
        system_msg = example.get("instruction", "You are a Rust programming assistant.")
        messages.append({"role": "system", "content": system_msg})

        # User message from input
        user_content = example.get("input", "")
        messages.append({"role": "user", "content": user_content})

        # Assistant message from output
        assistant_content = example.get("output", "")
        messages.append({"role": "assistant", "content": assistant_content})

        # Format using chat template
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Process all examples
    formatted = []
    for example in examples:
        text = format_example(example)
        if text:
            formatted.append({"text": text})

    logger.info(f"Prepared {len(formatted)} training examples")
    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for sleepy-coder")
    parser.add_argument("--config", "-c", help="Path to config YAML")
    parser.add_argument("--steps", "-s", type=int, help="Max training steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--data", "-d", help="Path to training data JSONL (use mixed.jsonl for replay)")
    parser.add_argument("--lora-r", type=int, help="LoRA rank (default: 8)")
    parser.add_argument("--base-model", "-m", help="Base model path (local or HF). Use to continue from merged model.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.steps:
        config.max_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.output:
        config.output_dir = args.output
    if args.data:
        config.data_path = args.data
    if args.lora_r:
        config.lora_r = args.lora_r
        config.lora_alpha = args.lora_r * 2  # Keep alpha = 2 * r
    if args.base_model:
        config.base_model = args.base_model
    if args.wandb:
        config.use_wandb = True

    # Resolve paths - support both absolute and relative paths
    # Relative paths are resolved from the project root (parent of cuda/)
    script_dir = Path(__file__).parent.parent  # cuda/
    project_root = script_dir.parent  # sleepy-coder/

    data_path = Path(config.data_path)
    if not data_path.is_absolute():
        data_path = (project_root / config.data_path).resolve()

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / config.output_dir).resolve()

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Sleepy Coder CUDA Training ===")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {run_dir}")
    logger.info(f"Steps: {config.max_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")

    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be slow.")
    else:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer (offline - no HuggingFace network requests)
    logger.info(f"Loading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config
    bnb_config = None
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # Load model (offline - no HuggingFace network requests)
    logger.info(f"Loading model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",  # Use Flash SDPA
    )
    model.config.use_cache = False

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    examples = load_sft_data(data_path)
    if not examples:
        logger.error(f"No training data found at {data_path}")
        logger.info("Generate training data with:")
        logger.info("  cd ../rust && cargo build --release")
        logger.info("  ./target/release/sleepy-coder export --output ../data/sft/train.jsonl")
        sys.exit(1)

    dataset = prepare_sft_dataset(examples, tokenizer)

    # Training arguments
    sft_config = SFTConfig(
        output_dir=str(run_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name=run_id if config.use_wandb else None,
        max_length=config.max_seq_length,
        dataset_text_field="text",
        gradient_checkpointing=config.gradient_checkpointing,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save adapter
    adapter_dir = run_dir / "adapter"
    logger.info(f"Saving adapter to {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    # Save metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples": len(dataset),
        "config": {
            "base_model": config.base_model,
            "max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training complete!")
    logger.info(f"Final loss: {train_result.training_loss:.4f}")
    logger.info(f"Adapter saved to: {adapter_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Merge adapter: python scripts/merge.py --adapter {adapter_dir}")
    logger.info(f"  2. Create Ollama model: ollama create sleepy-coder-v2 -f <modelfile>")
    logger.info(f"  3. Run eval: ./rust/target/release/sleepy-coder eval --cycle 2 --model sleepy-coder-v2")


if __name__ == "__main__":
    main()
