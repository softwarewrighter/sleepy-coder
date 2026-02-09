# CUDA Training Scripts

Scripts for training, merging, and exporting LoRA adapters on NVIDIA GPUs.

## Quick Reference

| Script | Purpose | Example |
|--------|---------|---------|
| `quick_test.py` | Validate CUDA setup | `python quick_test.py` |
| `train.py` | Train LoRA adapter | `python train.py --steps 100` |
| `merge.py` | Merge adapter → GGUF → Ollama | `python merge.py --adapter ../runs/adapters/XXX/adapter` |
| `continue_training.py` | Resume from checkpoint | `python continue_training.py --checkpoint ../runs/adapters/XXX/checkpoint-50` |
| `sync_from_mac.sh` | Copy data from Mac | `./sync_from_mac.sh mike@mac` |

---

## Script Details

### `quick_test.py`

Validates that CUDA training works before running full training.

**Checks:**
1. CUDA availability and GPU info
2. Model loading with 4-bit quantization
3. LoRA training loop (10 steps)

**Usage:**
```bash
source ../.venv/bin/activate
python quick_test.py
```

**Expected output:**
```
=== CUDA Check ===
CUDA available: True
GPU 0: NVIDIA GeForce RTX 3090
  Memory: 24.0 GB
...
=== Quick Test Results ===
  cuda: PASS
  model_loading: PASS
  lora_training: PASS
```

---

### `train.py`

Main training script for LoRA adapters.

**Features:**
- 4-bit QLoRA quantization
- Flash Attention (SDPA)
- Gradient checkpointing
- Configurable hyperparameters
- Offline mode (no HuggingFace network calls)

**Usage:**
```bash
# Basic training
python train.py --steps 100

# With custom data
python train.py --data ../data/sft/eval_aligned.jsonl --steps 200

# With custom hyperparameters
python train.py --steps 100 --lr 1e-4 --batch-size 2 --lora-r 8

# With config file
python train.py --config ../configs/full_train.yaml
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 100 | Training steps |
| `--lr` | 1e-4 | Learning rate |
| `--batch-size` | 4 | Batch size |
| `--lora-r` | 8 | LoRA rank |
| `--data` | `../data/sft/train.jsonl` | Training data path |

**Output:**
```
runs/adapters/<timestamp>/
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoint-50/
├── checkpoint-100/
└── metrics.json
```

---

### `merge.py`

Merges LoRA adapter into base model and exports to Ollama.

**Pipeline:**
1. Load base model + adapter
2. Merge adapter weights into base
3. Convert to GGUF format
4. Quantize to q4_k_m
5. Create Ollama model

**Usage:**
```bash
# Full pipeline (merge + quantize + ollama)
python merge.py --adapter ../runs/adapters/20260209_123456/adapter

# Custom model name
python merge.py --adapter ../runs/adapters/XXX/adapter --model-name sleepy-coder-v14

# Skip Ollama creation
python merge.py --adapter ../runs/adapters/XXX/adapter --skip-ollama

# Different quantization
python merge.py --adapter ../runs/adapters/XXX/adapter --quantize q8_0
```

**Requirements:**
- llama.cpp (for GGUF conversion and quantization)
- Ollama (for model creation)

---

### `continue_training.py`

Resume training from an existing checkpoint.

**Usage:**
```bash
# Continue for 500 more steps
python continue_training.py --checkpoint ../runs/adapters/XXX/checkpoint-50 --steps 500

# With custom learning rate
python continue_training.py --checkpoint ../runs/adapters/XXX/checkpoint-50 --steps 500 --lr 5e-5
```

---

### `sync_from_mac.sh`

Syncs training data and checkpoints from Mac to Linux.

**Usage:**
```bash
./sync_from_mac.sh mike@mikes-macbook.local
```

**Syncs:**
- `data/sft/` → Training data
- `rust/data/episodes/` → Eval episodes
- `runs/adapters/` → Checkpoints (optional)

---

## Training Data Format

Scripts expect JSONL files with this format:

```json
{
  "instruction": "You are a Rust expert. Fix the following code...",
  "input": "## Buggy Code:\n```rust\nfn main() { let s = String::new(); let t = s; println!(\"{}\", s); }\n```\n\n## Compiler Error:\nUse after move\n\n## Fixed Code:",
  "output": "fn main() { let s = String::new(); let t = s.clone(); println!(\"{}\", s); }",
  "task_id": "bc_001"
}
```

---

## Hyperparameter Recommendations

### To Prevent Catastrophic Forgetting

| Parameter | Aggressive | Conservative | Why |
|-----------|------------|--------------|-----|
| LoRA rank | 16 | **8** | Lower rank = less capacity to overwrite |
| Learning rate | 2e-4 | **1e-4** | Slower updates = less forgetting |
| Steps | 500 | **100** | Shorter cycles with eval gates |
| Dropout | 0.05 | **0.1** | More regularization |

### By GPU Memory

| VRAM | Batch Size | Seq Length | Notes |
|------|------------|------------|-------|
| 8GB | 1 | 1024 | Tight, use gradient accumulation |
| 12GB | 2 | 2048 | Comfortable |
| 16GB | 4 | 2048 | Recommended |
| 24GB | 8 | 2048 | Fast training |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch-size 1 --steps 100

# Or increase gradient accumulation in config
# gradient_accumulation_steps: 4
```

### Model Loading Fails

```bash
# Ensure base model is downloaded
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')"
```

### GGUF Conversion Fails

```bash
# Build llama.cpp
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
mkdir build && cd build
cmake ..
make llama-quantize
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. SETUP                                                   │
│     ./setup.sh                                              │
│     python quick_test.py                                    │
│                                                             │
│  2. PREPARE DATA                                            │
│     python ../scripts/generate_eval_aligned_koans.py        │
│                                                             │
│  3. TRAIN                                                   │
│     python train.py --data ../data/sft/eval_aligned.jsonl   │
│              ↓                                              │
│     runs/adapters/<timestamp>/adapter/                      │
│                                                             │
│  4. EXPORT                                                  │
│     python merge.py --adapter runs/adapters/XXX/adapter     │
│              ↓                                              │
│     Ollama model: sleepy-coder-vN                           │
│                                                             │
│  5. EVALUATE                                                │
│     ./rust/target/release/sleepy-coder eval \               │
│         --cycle N --model sleepy-coder-vN                   │
│              ↓                                              │
│     Pass rate comparison vs baseline                        │
│                                                             │
│  6. ITERATE                                                 │
│     If improved: deploy                                     │
│     If not: adjust data/hyperparameters, goto 2             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## See Also

- [docs/next-steps.md](../../docs/next-steps.md) — Analysis and recommendations
- [docs/course-correction.md](../../docs/course-correction.md) — Share implementation details
- [docs/changes.md](../../docs/changes.md) — Training history and learnings
- [cuda/docs/workflow.html](../docs/workflow.html) — Interactive workflow diagram
