# Sleepy Coder CUDA Training

This directory contains scripts for training sleepy-coder on a Linux workstation with NVIDIA GPU.

## Quick Start

```bash
# 1. Clone repo on Linux box
git clone <repo-url>
cd sleepy-coder/cuda

# 2. Sync training data from Mac (run ON LINUX)
chmod +x scripts/sync_from_mac.sh
./scripts/sync_from_mac.sh mike@mikes-macbook.local

# 3. Run setup
chmod +x setup.sh
./setup.sh

# 4. Activate and validate
source .venv/bin/activate
python scripts/quick_test.py

# 5. If tests pass, run full training
python scripts/train.py --steps 500
```

## Syncing Files from Mac

The training data and any existing checkpoints need to be copied from Mac:

```bash
# Option 1: Use the sync script (recommended)
./scripts/sync_from_mac.sh mike@mac-hostname

# Option 2: Manual scp
scp -r mike@mac:~/github/softwarewrighter/sleepy-coder/data/sft ./data/
scp -r mike@mac:~/github/softwarewrighter/sleepy-coder/runs/adapters ./runs/

# Option 3: Continue from Mac checkpoint
scp -r mike@mac:~/github/softwarewrighter/sleepy-coder/runs/adapters/20260208_163226 ./runs/adapters/
python scripts/continue_training.py --checkpoint ./runs/adapters/20260208_163226/checkpoint-50 --steps 500
```

## Prerequisites

- **Linux** with NVIDIA GPU (8GB+ VRAM recommended)
- **NVIDIA Driver** 525+
- **CUDA Toolkit** 11.8 or 12.x
- **Python** 3.10+
- **Ollama** (for evaluation)
- **Rust** (optional, for eval harness)

## Directory Structure

```
cuda/
├── README.md           # This file
├── setup.sh            # One-time setup script
├── requirements.txt    # Python dependencies
├── configs/
│   ├── quick_test.yaml # Minimal test config (20 steps)
│   └── full_train.yaml # Full training config (500 steps)
├── scripts/
│   ├── quick_test.py   # Validates CUDA/training works
│   ├── train.py        # Main training script
│   └── merge.py        # Merge adapter + export to Ollama
└── .venv/              # Virtual environment (created by setup.sh)
```

## Step-by-Step Instructions

### 1. Initial Setup

Run the setup script once after cloning:

```bash
cd cuda
chmod +x setup.sh
./setup.sh
```

This will:
- Check for NVIDIA GPU
- Create Python virtual environment
- Install dependencies
- Download base model (Qwen2.5-Coder-1.5B-Instruct)
- Install Ollama (if not present)
- Pull Ollama base model
- Build Rust CLI (if cargo available)

### 2. Validate Setup

Run the quick test to verify everything works:

```bash
source .venv/bin/activate
python scripts/quick_test.py
```

Expected output:
```
=== CUDA Check ===
PyTorch version: 2.x.x
CUDA version: 12.x
GPU count: 1
GPU 0: NVIDIA GeForce RTX 3090
  Memory: 24.0 GB
  Compute: 8.6
...
=== Quick Test Results ===
  cuda: PASS
  model_loading: PASS
  lora_training: PASS

All tests passed! Ready for full training.
```

### 3. Training

#### Quick Validation Run (20 steps, ~2-5 min)

```bash
python scripts/train.py --config configs/quick_test.yaml
```

#### Full Training (500 steps, ~30-60 min)

```bash
python scripts/train.py --config configs/full_train.yaml
```

Or with custom parameters:

```bash
python scripts/train.py --steps 500 --lr 2e-4 --batch-size 4
```

Training output will be saved to `../runs/adapters/<timestamp>/`.

### 4. Export to Ollama

After training, merge the adapter and create an Ollama model:

```bash
# Find your adapter
ls ../runs/adapters/

# Merge and export (auto-creates Ollama model)
python scripts/merge.py --adapter ../runs/adapters/20260209_123456/adapter

# Or with custom name
python scripts/merge.py --adapter ../runs/adapters/20260209_123456/adapter --model-name sleepy-coder-v2
```

### 5. Evaluate

Run the evaluation harness:

```bash
# Build Rust CLI if not already built
cd ../rust
cargo build --release

# Run evaluation
./target/release/sleepy-coder eval --cycle 2 --model sleepy-coder-v2
```

## Training Data

Training data comes from `../data/episodes/`. Each JSONL file contains episodes from evaluation runs.

If no data exists, first run baseline evaluation:

```bash
cd ../rust
cargo build --release
./target/release/sleepy-coder eval --cycle 0
```

This generates episodes that can be used for training.

## Troubleshooting

### CUDA not available

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

If PyTorch doesn't see CUDA, reinstall with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

Reduce batch size:
```bash
python scripts/train.py --batch-size 1 --steps 500
```

Or use gradient accumulation:
```bash
# In config: gradient_accumulation_steps: 4
```

### Ollama issues

```bash
# Check Ollama is running
ollama list

# Restart if needed
sudo systemctl restart ollama
# or
ollama serve
```

## Expected Results

| Steps | Duration (RTX 3090) | Expected Improvement |
|-------|---------------------|---------------------|
| 20    | ~2-5 min            | Validation only     |
| 100   | ~10-15 min          | Minimal             |
| 500   | ~30-60 min          | 5-15% pass rate     |
| 1000  | ~1-2 hours          | 10-25% pass rate    |

## Monitoring

Watch training progress:

```bash
# Loss should decrease over time
tail -f ../runs/adapters/<run_id>/trainer_state.json
```

With Weights & Biases (optional):

```bash
wandb login
python scripts/train.py --wandb --steps 500
```

## Next Steps After Training

1. **Evaluate** - Compare trained model to baseline
2. **Iterate** - If improvement seen, train longer or tune hyperparameters
3. **More data** - Generate more training examples from failed evaluations
4. **Scale up** - Try larger models (7B, 14B) if resources allow
