"""LoRA training with PEFT."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from sleepy_pact.train.config import TrainConfig

logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA trainer for code fixing models."""

    def __init__(self, config: TrainConfig):
        """Initialize trainer with config."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def load_model(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading model: {self.config.base_model}")

        # Quantization config for memory efficiency
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
        )

        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Prepare for training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Model loaded successfully")

    def setup_lora(self):
        """Configure LoRA adapter."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    def prepare_dataset(self, examples: list[dict]) -> Dataset:
        """Prepare dataset for training.

        Args:
            examples: List of SFT examples with 'instruction', 'input', 'output'.

        Returns:
            HuggingFace Dataset ready for training.
        """

        def format_example(example: dict) -> str:
            """Format example for training."""
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            # Chat format
            text = f"""<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
            return text

        def tokenize(example: dict) -> dict:
            text = format_example(example)
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = Dataset.from_list(examples)
        tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
        return tokenized_dataset

    def train(
        self,
        train_data: list[dict],
        eval_data: list[dict] | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            train_data: Training examples.
            eval_data: Optional evaluation examples.
            run_name: Optional run name for output directory.

        Returns:
            Training metrics.
        """
        if self.model is None:
            self.load_model()

        if self.peft_model is None:
            self.setup_lora()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        eval_dataset = self.prepare_dataset(eval_data) if eval_data else None

        # Output directory
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.output_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            bf16=True,
            optim="adamw_torch",
            seed=self.config.seed,
            report_to=[],  # Disable wandb etc.
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save adapter (use safetensors for security - no pickle exploits)
        self.peft_model.save_pretrained(
            output_dir / "adapter",
            safe_serialization=self.config.save_safetensors,
        )
        self.tokenizer.save_pretrained(output_dir / "adapter")

        # Save metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples": len(train_dataset),
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Adapter saved to {output_dir}")
        return metrics


def train_lora(
    sft_data_path: Path,
    output_dir: Path,
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    """Convenience function to train LoRA from SFT data file.

    Args:
        sft_data_path: Path to JSONL file with SFT examples.
        output_dir: Directory to save adapter.
        config: Optional training config.

    Returns:
        Training metrics.
    """
    if config is None:
        config = TrainConfig()

    config.output_dir = output_dir

    # Load SFT data
    examples = []
    with open(sft_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    logger.info(f"Loaded {len(examples)} training examples")

    # Train
    trainer = LoRATrainer(config)
    metrics = trainer.train(examples)

    return metrics
