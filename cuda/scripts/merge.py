#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and convert to GGUF for Ollama.

Usage:
    python merge.py --adapter ../runs/adapters/20260208_123456/adapter
    python merge.py --adapter ../runs/adapters/latest/adapter --quantize q4_k_m
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def merge_adapter(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
) -> None:
    """Merge LoRA adapter into base model."""

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quantize: str = None,
) -> Path:
    """Convert HuggingFace model to GGUF format."""

    # Find convert script - prefer llama.cpp's version which is more up-to-date
    llama_cpp_converter = Path("/tmp/llama.cpp/convert_hf_to_gguf.py")
    script_dir = Path(__file__).parent.parent
    local_convert_script = script_dir / "scripts" / "convert_hf_to_gguf.py"

    if llama_cpp_converter.exists():
        convert_script = llama_cpp_converter
        logger.info(f"Using llama.cpp converter: {convert_script}")
    elif local_convert_script.exists():
        convert_script = local_convert_script
        logger.info(f"Using local converter: {convert_script}")
    else:
        # Clone llama.cpp to get the converter
        logger.info("Cloning llama.cpp to get converter...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", "/tmp/llama.cpp"], check=True)
        convert_script = llama_cpp_converter

    # Convert to FP16 GGUF first
    fp16_gguf = output_path / "model-f16.gguf"
    logger.info(f"Converting to GGUF: {fp16_gguf}")

    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(model_path),
            "--outfile", str(fp16_gguf),
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"GGUF conversion failed: {result.stderr}")
        raise RuntimeError("GGUF conversion failed")

    logger.info(f"Created: {fp16_gguf}")

    # Quantize if requested
    if quantize:
        quantized_gguf = output_path / f"model-{quantize.lower()}.gguf"
        logger.info(f"Quantizing to {quantize}: {quantized_gguf}")

        # Find llama-quantize binary
        llama_quantize = Path("/tmp/llama.cpp/build/bin/llama-quantize")
        if not llama_quantize.exists():
            # Try to find in PATH
            import shutil
            llama_quantize = shutil.which("llama-quantize")
            if llama_quantize is None:
                logger.error("llama-quantize not found. Build llama.cpp first:")
                logger.error("  cd /tmp/llama.cpp && mkdir build && cd build && cmake .. && make llama-quantize")
                raise RuntimeError("llama-quantize not found")

        result = subprocess.run(
            [str(llama_quantize), str(fp16_gguf), str(quantized_gguf), quantize.upper()],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            raise RuntimeError("Quantization failed")

        logger.info(f"Created: {quantized_gguf}")
        return quantized_gguf

    return fp16_gguf


def create_modelfile(gguf_path: Path, output_path: Path) -> Path:
    """Create Ollama Modelfile."""

    modelfile = output_path / "Modelfile"

    content = f"""FROM {gguf_path.name}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""

    modelfile.write_text(content)
    logger.info(f"Created: {modelfile}")
    return modelfile


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export to Ollama")
    parser.add_argument("--adapter", "-a", required=True, help="Path to adapter directory")
    parser.add_argument("--base-model", "-b", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output", "-o", help="Output directory (default: ../runs/merged/<run_id>)")
    parser.add_argument("--quantize", "-q", default="q4_k_m", help="Quantization type (q4_k_m, q8_0, etc.)")
    parser.add_argument("--model-name", "-n", help="Ollama model name (default: sleepy-coder-<run_id>)")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama model creation")
    args = parser.parse_args()

    adapter_path = Path(args.adapter).resolve()
    if not adapter_path.exists():
        logger.error(f"Adapter not found: {adapter_path}")
        sys.exit(1)

    # Determine run ID from adapter path
    run_id = adapter_path.parent.name
    if run_id == "adapter":
        run_id = adapter_path.parent.parent.name

    # Output directory
    script_dir = Path(__file__).parent.parent
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = (script_dir / "runs" / "merged" / run_id).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Merge and Export ===")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Quantize: {args.quantize}")

    # Step 1: Merge adapter
    merged_path = output_dir / "hf_model"
    merge_adapter(args.base_model, adapter_path, merged_path)

    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(merged_path, output_dir, args.quantize)

    # Step 3: Create Modelfile
    modelfile = create_modelfile(gguf_path, output_dir)

    # Step 4: Create Ollama model
    if not args.skip_ollama:
        model_name = args.model_name or f"sleepy-coder-{run_id}"
        logger.info(f"Creating Ollama model: {model_name}")

        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile)],
            cwd=output_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Ollama create failed: {result.stderr}")
        else:
            logger.info(f"Ollama model created: {model_name}")

    logger.info("")
    logger.info("=== Export Complete ===")
    logger.info(f"Merged model: {merged_path}")
    logger.info(f"GGUF: {gguf_path}")
    logger.info(f"Modelfile: {modelfile}")
    if not args.skip_ollama:
        logger.info(f"Ollama model: {model_name}")
        logger.info("")
        logger.info("To evaluate:")
        logger.info(f"  ./rust/target/release/sleepy-coder eval --cycle 2 --model {model_name}")


if __name__ == "__main__":
    main()
