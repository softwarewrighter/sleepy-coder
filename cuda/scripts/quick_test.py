#!/usr/bin/env python3
"""
Quick validation test for CUDA training setup.

This script runs a minimal training loop (10 steps) to verify:
1. CUDA is working
2. Model loads correctly
3. Training loop runs
4. Checkpoints save properly

Usage:
    python quick_test.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_cuda():
    """Verify CUDA is available and working."""
    logger.info("=== CUDA Check ===")

    if not torch.cuda.is_available():
        logger.error("CUDA is NOT available!")
        logger.info("Possible issues:")
        logger.info("  - NVIDIA driver not installed")
        logger.info("  - PyTorch not built with CUDA support")
        logger.info("  - CUDA toolkit version mismatch")
        return False

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        logger.info(f"  Compute: {props.major}.{props.minor}")

    # Quick tensor test
    logger.info("Testing CUDA tensor operations...")
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    logger.info(f"Matrix multiply (1000x1000): {elapsed*1000:.1f}ms")

    return True


def check_model_loading():
    """Verify model can be loaded with quantization."""
    logger.info("")
    logger.info("=== Model Loading Check ===")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Vocab size: {tokenizer.vocab_size}")

        logger.info(f"Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",  # Use Flash SDPA
        )

        # Check model is on GPU
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")

        # Quick inference test
        logger.info("Testing inference...")
        inputs = tokenizer("fn main() {", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated: {result[:50]}...")

        return True, tokenizer, model

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False, None, None


def check_lora_training(tokenizer, model):
    """Verify LoRA training works."""
    logger.info("")
    logger.info("=== LoRA Training Check (10 steps) ===")

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig

        # Prepare for training
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=8,  # Small for quick test
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Minimal dataset
        examples = [
            {"text": "<|im_start|>user\nFix: fn main() { let x = 5; }<|im_end|>\n<|im_start|>assistant\nfn main() { let x: i32 = 5; }<|im_end|>"},
            {"text": "<|im_start|>user\nFix: fn add(a, b) { a + b }<|im_end|>\n<|im_start|>assistant\nfn add(a: i32, b: i32) -> i32 { a + b }<|im_end|>"},
        ]
        dataset = Dataset.from_list(examples)

        # Output dir
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "runs" / "quick_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=str(output_dir),
            max_steps=10,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            logging_steps=2,
            save_steps=10,
            bf16=True,
            optim="paged_adamw_32bit",
            report_to="none",
            max_length=256,
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting 10-step training...")
        start = time.time()
        result = trainer.train()
        elapsed = time.time() - start

        logger.info(f"Training complete in {elapsed:.1f}s")
        logger.info(f"Final loss: {result.training_loss:.4f}")

        # Save test checkpoint
        trainer.model.save_pretrained(output_dir / "adapter", safe_serialization=True)
        logger.info(f"Checkpoint saved to: {output_dir / 'adapter'}")

        return True

    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("=" * 50)
    logger.info("Sleepy Coder CUDA Quick Test")
    logger.info("=" * 50)
    logger.info("")

    results = {}

    # Test 1: CUDA
    results["cuda"] = check_cuda()
    if not results["cuda"]:
        logger.error("CUDA check failed. Cannot continue.")
        sys.exit(1)

    # Test 2: Model loading
    success, tokenizer, model = check_model_loading()
    results["model_loading"] = success
    if not success:
        logger.error("Model loading failed. Cannot continue.")
        sys.exit(1)

    # Test 3: LoRA training
    results["lora_training"] = check_lora_training(tokenizer, model)

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Quick Test Results")
    logger.info("=" * 50)

    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {test}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("")
        logger.info("All tests passed! Ready for full training.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run full training: python scripts/train.py --steps 500")
        logger.info("  2. Merge and export: python scripts/merge.py --adapter <path>")
        logger.info("  3. Evaluate: ./rust/target/release/sleepy-coder eval --cycle 2")
    else:
        logger.error("")
        logger.error("Some tests failed. Please fix issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
