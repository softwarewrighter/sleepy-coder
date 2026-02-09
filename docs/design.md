# Technical Design: Sleepy Coder

## Design Principles

1. **Local-first**: Run on consumer hardware with small models
2. **Reproducible**: Fixed seeds, versioned artifacts, deterministic evaluation
3. **Safe**: Never regress, always gate, always rollback on failure
4. **Simple**: Minimal agent loop, clear data flow, avoid over-engineering
5. **Demonstrable**: Visual proof of learning for videos/demos

## Key Technical Decisions

### 1. Model Selection

**Target**: 1B-3B parameter models with 4-bit quantization

Recommended models:
- `qwen2.5-coder:1.5b-instruct-q4_K_M` (Ollama)
- `deepseek-coder:1.3b-instruct-q4_K_M` (Ollama)
- `codellama:7b-instruct-q4_K_M` (if more VRAM available)

Rationale:
- Small enough for local GPU (4-8GB VRAM)
- Good enough at code to show improvement
- Fast inference for tight demo loops
- Imperfect baseline = room to show learning

### 2. Error Signature Normalization

To detect repeat errors, normalize compiler output:

```rust
pub fn normalize_error(raw: &str) -> String {
    // 1. Extract error code (e.g., E0382)
    // 2. Extract error type (e.g., "borrow of moved value")
    // 3. Remove file paths and line numbers
    // 4. Remove variable names (replace with placeholders)
    // 5. Hash the result
}
```

Example:
```
Raw: "error[E0382]: borrow of moved value: `s`"
Normalized: "E0382:borrow_of_moved_value"
```

### 3. Training Data Format

Supervised fine-tuning pairs:

```json
{
  "input": "<|system|>You are a Rust coding assistant.<|end|>\n<|user|>Fix this code:\n```rust\nfn main() { let s = String::new(); let t = s; println!(\"{}\", s); }\n```\nError: borrow of moved value: `s`<|end|>\n<|assistant|>",
  "output": "```rust\nfn main() { let s = String::new(); let t = s.clone(); println!(\"{}\", s); }\n```<|end|>"
}
```

### 4. LoRA Configuration

**Updated 2026-02-08**: Conservative settings to prevent catastrophic forgetting.

```python
lora_config = LoraConfig(
    r=8,                    # Low rank = less forgetting
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,       # Higher dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=100,          # Shorter cycles, more frequent eval
    learning_rate=1e-4,     # Lower LR = less forgetting
    bf16=True,              # BF16 for modern GPUs
    logging_steps=10,
    output_dir=output_dir,
)
```

### 4a. Data Preparation (Critical)

**Research shows replay is essential.** Prepare mixed dataset before training:

```python
# scripts/prepare_training_data.py
dataset_composition = {
    "replay": 0.44,          # Original training data
    "success": 0.42,         # Tasks that passed
    "hard_cases": 0.14,      # Failed tasks (extra copies)
}
```

This mix prevents the model from forgetting base capabilities while learning from failures.

### 5. Share-Style Consolidation (Phase 2) - REVISED

**Status**: Not yet properly implemented. See [course-correction.md](./course-correction.md).

#### Share Algorithm (arXiv:2602.06043)

**Phase 1: Initialization**
```python
def share_consolidate(adapters: List[Path], k_target_variance: float = 0.6):
    """Extract shared subspace from N adapters trained on distinct tasks."""
    # 1. Load LoRA adapters
    loras = [load_lora_adapter(p) for p in adapters]

    # 2. For each layer, stack delta_W matrices
    deltas = {}
    for layer in loras[0].keys():
        deltas[layer] = np.stack([
            (lora[layer]['B'] @ lora[layer]['A']).flatten()
            for lora in loras
        ])

    # 3. Center by subtracting mean
    for layer in deltas:
        deltas[layer] -= deltas[layer].mean(axis=0)

    # 4. SVD to find principal directions
    basis = {}
    for layer in deltas:
        U, S, Vh = np.linalg.svd(deltas[layer], full_matrices=False)

        # Select k for 60% explained variance
        cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
        k = np.searchsorted(cumvar, k_target_variance) + 1

        basis[layer] = {'Vh': Vh[:k], 'k': k}  # FROZEN

    # 5. Compute per-adapter coefficients
    coefficients = []
    for lora in loras:
        coef = {}
        for layer in basis:
            delta = (lora[layer]['B'] @ lora[layer]['A']).flatten()
            coef[layer] = basis[layer]['Vh'] @ delta
        coefficients.append(coef)

    return SharedBasis(basis, coefficients)
```

**Phase 2: Continual Adaptation (Coefficient-Only Training)**
```python
def train_coefficients(
    shared_basis: SharedBasis,
    new_data: Dataset,
    base_model: str,
) -> Coefficients:
    """Train ONLY coefficients for new task (basis frozen)."""
    # Initialize coefficients
    coefficients = {
        layer: torch.zeros(shared_basis.basis[layer]['k'], requires_grad=True)
        for layer in shared_basis.basis
    }

    # Training loop - MUCH cheaper than full LoRA
    for batch in new_data:
        # Reconstruct delta_W from basis + coefficients
        # Forward pass, compute loss, backprop through coefficients only
        ...

    return coefficients
```

**Phase 3: Merging (Incremental Subspace Update)**
```python
def merge_new_adapter(shared_basis: SharedBasis, new_adapter: LoraAdapter):
    """Merge new adapter into shared subspace."""
    # 1. Reconstruct all prior adapters from basis + coefficients
    reconstructed = [
        reconstruct_adapter(shared_basis.basis, coef)
        for coef in shared_basis.coefficients
    ]

    # 2. Add new adapter
    reconstructed.append(new_adapter)

    # 3. Re-run SVD to update subspace
    return share_consolidate(reconstructed)
```

#### Key Hyperparameters (from paper)
- **k**: Principal factors at 60% explained variance threshold
- **p=1**: Pseudo-rank is effective; higher values yield minimal benefit
- **φ=[1, k/4]**: Temporary factors range
- **Number of adapters**: 10-50+ for good subspace estimation

#### What We Did Wrong
1. Only trained 6 adapters on similar data (need 10-50 on distinct tasks)
2. Just merged weights (need SVD basis extraction)
3. Retrained full LoRA each cycle (need coefficient-only training)
4. Used 50%+ replay (Share avoids replay entirely)

### 6. UWSH Coefficient Updates (Phase 2)

Based on Universal Weight Subspace Hypothesis (arXiv:2512.05117):

The key insight: adapters trained on related tasks share a common low-rank subspace.
Once we have the shared basis, new tasks only need to learn coefficients.

**Compression ratio**: 50 adapters × 4MB each = 200MB → 50 coefficient vectors × ~1KB = 50KB

### 7. Evaluation Gates

```rust
struct EvalGates {
    min_pass_rate: f64,           // e.g., 0.90
    max_regression_delta: f64,    // e.g., 0.02
    max_repeat_rate_increase: f64, // e.g., 0.0 (must not increase)
}

fn check_gates(
    current: &EvalResult,
    previous: &EvalResult,
    gates: &EvalGates,
) -> GateResult {
    if current.pass_rate < gates.min_pass_rate {
        return GateResult::Fail("pass rate too low");
    }
    if current.pass_rate < previous.pass_rate - gates.max_regression_delta {
        return GateResult::Fail("regression detected");
    }
    if current.repeat_error_rate > previous.repeat_error_rate {
        return GateResult::Fail("repeat rate increased");
    }
    GateResult::Pass
}
```

### 8. Quick Cycle for Demos

`scripts/quick_cycle.sh`:

```bash
#!/bin/bash
set -euo pipefail

N_CYCLES=${1:-6}
N_TASKS_DAY=${2:-10}
N_STEPS_TRAIN=${3:-300}

for i in $(seq 1 $N_CYCLES); do
    echo "=== Cycle $i ==="

    # Day: run agent on tasks
    cargo run -p cli -- run-day --n $N_TASKS_DAY

    # Sleep: train on captured episodes
    python -m sleepy_pact.train.lora_train --max-steps $N_STEPS_TRAIN

    # Eval: measure improvement
    cargo run -p cli -- eval

    # Plot: visualize progress
    python -m sleepy_pact.viz.plots
done
```

### 9. Visualization Design

Three key plots for demos:

**Plot 1: Repeat Error Rate**
- Y-axis: Fraction of tasks triggering same error signature
- X-axis: Sleep cycle (0, 1, 2, ...)
- Expected: Decreasing trend

**Plot 2: Steps to Green**
- Y-axis: Median tool calls until pass
- X-axis: Sleep cycle
- Expected: Decreasing trend

**Plot 3: Pass Rate on Frozen Set**
- Y-axis: Percentage of frozen tasks passing
- X-axis: Sleep cycle
- Expected: Stable or increasing (never decreasing)

### 10. CLI Design

```
sleepy-coder CLI

USAGE:
    sleepy-coder <COMMAND>

COMMANDS:
    init        Initialize a new run directory
    run-day     Run agent on tasks, capture episodes
    sleep       Train adapter from captured episodes
    eval        Evaluate model on frozen suite
    plot        Generate visualization PNGs
    cycle       Run full day-sleep-eval-plot cycle
    status      Show current run status
    rollback    Rollback to previous adapter
```

## File Formats

### config.yaml

```yaml
run_id: "2026-02-08_01"
model:
  backend: ollama
  name: qwen2.5-coder:1.5b-instruct-q4_K_M

day:
  n_tasks: 30
  max_attempts: 5

sleep:
  max_steps: 500
  batch_size: 4
  lora_rank: 8

eval:
  frozen_set: data/evalsets/frozen_v1.jsonl
  gates:
    min_pass_rate: 0.85
    max_regression_delta: 0.05
```

### Episode Schema (SQLite)

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_idx INTEGER NOT NULL,
    prompt_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    error_signature TEXT,
    diff_unified TEXT,
    passed BOOLEAN NOT NULL,
    steps_to_green INTEGER,
    wall_clock_ms INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_run_task ON episodes(run_id, task_id);
CREATE INDEX idx_error_sig ON episodes(error_signature);
```

### Metrics JSONL

```json
{"cycle": 0, "repeat_error_rate": 0.45, "median_steps_to_green": 4.0, "pass_rate": 0.70, "frozen_pass_rate": 0.85}
{"cycle": 1, "repeat_error_rate": 0.38, "median_steps_to_green": 3.5, "pass_rate": 0.75, "frozen_pass_rate": 0.87}
{"cycle": 2, "repeat_error_rate": 0.30, "median_steps_to_green": 3.0, "pass_rate": 0.82, "frozen_pass_rate": 0.88}
```

## Error Handling

### Agent Errors
- Max attempts exceeded -> log episode as failed, continue to next task
- LLM timeout -> retry with backoff, then fail task
- Sandbox crash -> isolate, log, skip task

### Training Errors
- OOM -> reduce batch size, retry
- NaN loss -> skip batch, log warning
- Corrupted episodes -> validate on load, skip invalid

### Promotion Errors
- Gate failure -> keep previous adapter, log reason
- Rollback failure -> restore from versioned snapshot
