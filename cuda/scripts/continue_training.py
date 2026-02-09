#!/usr/bin/env python3
"""
Continue training from an existing checkpoint.

Usage:
    python continue_training.py --checkpoint ../runs/adapters/20260208_163226/checkpoint-50 --steps 500
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint directory")
    parser.add_argument("--steps", "-s", type=int, default=500, help="Additional steps to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Find base run directory
    run_dir = checkpoint_path.parent
    if checkpoint_path.name.startswith("checkpoint-"):
        run_dir = checkpoint_path.parent

    logger.info(f"=== Continue Training ===")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Additional steps: {args.steps}")

    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available!")

    # Load training args from checkpoint
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
        current_step = trainer_state.get("global_step", 0)
        logger.info(f"Resuming from step {current_step}")
    else:
        current_step = 0

    # Load tokenizer
    base_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model with quantization
    logger.info(f"Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Load adapter from checkpoint
    logger.info(f"Loading adapter from checkpoint...")
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    model.print_trainable_parameters()

    # Load training data
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "data" / "sft" / "train.jsonl"
    if not data_path.exists():
        data_path = script_dir.parent / "data" / "sft" / "train.jsonl"

    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        return

    logger.info(f"Loading training data from {data_path}")
    examples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Format as chat
    formatted = []
    for ex in examples:
        messages = [
            {"role": "system", "content": ex.get("instruction", "")},
            {"role": "user", "content": ex.get("input", "")},
            {"role": "assistant", "content": ex.get("output", "")},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    logger.info(f"Training examples: {len(dataset)}")

    # Output directory for continued training
    new_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_dir = script_dir / "runs" / "adapters" / f"continued_{new_run_id}"
    new_run_dir.mkdir(parents=True, exist_ok=True)

    # Training args
    total_steps = current_step + args.steps
    training_args = TrainingArguments(
        output_dir=str(new_run_dir),
        max_steps=total_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field="text",
    )

    # Resume training
    logger.info(f"Continuing training from step {current_step} to {total_steps}...")
    result = trainer.train(resume_from_checkpoint=str(checkpoint_path))

    # Save final adapter
    adapter_dir = new_run_dir / "adapter"
    logger.info(f"Saving adapter to {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    # Save metrics
    metrics = {
        "continued_from": str(checkpoint_path),
        "initial_step": current_step,
        "final_step": total_steps,
        "train_loss": result.training_loss,
        "timestamp": datetime.now().isoformat(),
    }
    with open(new_run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training complete!")
    logger.info(f"Final loss: {result.training_loss:.4f}")
    logger.info(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
