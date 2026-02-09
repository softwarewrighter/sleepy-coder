#!/bin/bash
# Setup script for CUDA training environment
# Run this on the Linux workstation with NVIDIA GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Sleepy Coder CUDA Setup ==="
echo "Script dir: $SCRIPT_DIR"
echo "Repo root: $REPO_ROOT"

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo ""
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "=== Creating virtual environment ==="
    python3 -m venv "$VENV_DIR"
fi

# Activate and install dependencies
echo ""
echo "=== Installing dependencies ==="
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# Verify CUDA is available to PyTorch
echo ""
echo "=== Verifying CUDA ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA not available! Training will be slow.')
"

# Download base model
echo ""
echo "=== Downloading base model ==="
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('Base model downloaded successfully!')
"

# Build Rust CLI if cargo is available
if command -v cargo &> /dev/null; then
    echo ""
    echo "=== Building Rust CLI ==="
    cd "$REPO_ROOT/rust"
    cargo build --release
    echo "Rust CLI built: $REPO_ROOT/rust/target/release/sleepy-coder"
else
    echo ""
    echo "NOTE: cargo not found. Rust CLI not built."
    echo "Install Rust from https://rustup.rs/ if you need the eval harness."
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "=== Installing Ollama ==="
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Pull base model for Ollama
echo ""
echo "=== Pulling Ollama base model ==="
ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To train:"
echo "  python $SCRIPT_DIR/scripts/train.py --steps 500"
echo ""
echo "To evaluate:"
echo "  $REPO_ROOT/rust/target/release/sleepy-coder eval --cycle 2"
