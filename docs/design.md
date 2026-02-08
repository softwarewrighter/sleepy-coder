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

```python
lora_config = LoraConfig(
    r=8,                    # Low rank for small models
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir=output_dir,
)
```

### 5. Share-Style Consolidation (Phase 2)

Algorithm 1 from the Share paper:

```python
def consolidate(adapters: List[LoraAdapter], rank_k: int):
    # 1. Collect LoRA delta_W matrices
    deltas = [compute_delta_w(a) for a in adapters]

    # 2. Stack and center
    stacked = np.stack([d.flatten() for d in deltas])
    centered = stacked - stacked.mean(axis=0)

    # 3. SVD to find shared subspace
    U, S, Vh = np.linalg.svd(centered, full_matrices=False)
    basis = Vh[:rank_k]  # Top-k principal directions

    # 4. Compute per-task coefficients
    coefficients = [project(d.flatten(), basis) for d in deltas]

    return SharedBasis(basis, coefficients)
```

### 6. UWSH Coefficient Updates (Phase 2)

Based on Universal Weight Subspace Hypothesis:

```python
def update_coefficients(
    frozen_basis: np.ndarray,
    new_examples: Dataset,
    current_coefficients: np.ndarray,
) -> np.ndarray:
    # Only update the small coefficient vector
    # Basis directions are frozen (reused across tasks)
    # Much cheaper than full LoRA training
    pass
```

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
