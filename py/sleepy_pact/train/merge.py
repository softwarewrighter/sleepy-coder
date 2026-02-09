"""Merge LoRA adapter into base model and export for Ollama."""

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def merge_and_export(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    push_to_hub: bool = False,
) -> None:
    """Merge LoRA adapter into base model and save.

    Args:
        base_model: Base model name or path.
        adapter_path: Path to LoRA adapter.
        output_path: Path to save merged model.
        push_to_hub: Whether to push to HuggingFace Hub.
    """
    logger.info(f"Loading base model: {base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading adapter from: {adapter_path}")

    # Load and merge adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")

    # Save merged model
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")

    if push_to_hub:
        logger.info("Pushing to HuggingFace Hub...")
        model.push_to_hub(output_path.name)
        tokenizer.push_to_hub(output_path.name)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", "-b", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", "-a", required=True, help="Path to adapter")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument("--push-to-hub", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    merge_and_export(
        base_model=args.base_model,
        adapter_path=Path(args.adapter),
        output_path=Path(args.output),
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
