# sleepy-pact

Python training and visualization for sleepy-coder.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
# Prepare SFT dataset from episodes
sleepy-pact prepare -e data/episodes/cycle_0.jsonl -t data/export/tasks.json -o data/sft/train.jsonl

# Train LoRA adapter
sleepy-pact train -d data/sft/train.jsonl -o runs/adapters

# Generate plots from metrics
sleepy-pact plot -m data/episodes/metrics.jsonl -o viz/

# Export adapter for Ollama
sleepy-pact export-ollama -a runs/adapters/latest
```

## Training on CUDA vs MLX

For prototyping, training can run on CPU or Apple Silicon (MLX).
For production training cycles, use a Linux box with NVIDIA GPU and CUDA.

The training uses safetensors format by default (no pickle security issues).
