# System Architecture: Sleepy Coder

## Overview

Sleepy Coder uses a dual-language architecture:
- **Rust**: Agent runtime, orchestration, capture, evaluation (fast, reliable, deterministic)
- **Python**: Training, embeddings, consolidation, visualization (ML ecosystem)

```
+-------------------+     +-------------------+     +-------------------+
|   DAY LOOP        |     |   SLEEP LOOP      |     |   EVAL LOOP       |
|   (Rust Agent)    | --> |   (Python Train)  | --> |   (Rust Eval)     |
+-------------------+     +-------------------+     +-------------------+
        |                         |                         |
        v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   episodes.sqlite |     |   adapter/        |     |   metrics.jsonl   |
|   (capture)       |     |   (LoRA weights)  |     |   (results)       |
+-------------------+     +-------------------+     +-------------------+
```

## Directory Structure

```
sleepy-coder/
  rust/
    Cargo.toml                 # workspace
    crates/
      core_types/              # shared structs, serde, schema versions
      capture/                 # episode storage, sqlite, diffs
      sandbox/                 # cargo check/test in temp dirs
      agent/                   # minimal Pi-like loop
      tasks_rust_koans/        # task generator/loader
      eval/                    # evaluation harness
      cli/                     # unified CLI wrapper

  py/
    pyproject.toml
    sleepy_pact/
      data/                    # schema, episode loading
      embed/                   # optional: vectorize for clustering
      curate/                  # cluster mistakes, build SFT dataset
      train/                   # LoRA training, Share merge, UWSH basis
      serve/                   # optional: inference server wrapper
      eval/                    # python-side eval helpers
      viz/                     # plots, dashboard

  data/
    koans/                     # frozen task definitions
    episodes/                  # exported JSONL backups
    sft/                       # training datasets
    evalsets/                  # frozen evaluation sets

  runs/
    <timestamp>/               # each run has config + outputs
      config.yaml
      day/episodes.sqlite
      sleep/adapter/
      eval/metrics.jsonl
      viz/*.png

  scripts/
    bootstrap.sh               # first-time setup
    run_day.sh                 # run agent on tasks
    run_sleep.sh               # train overnight
    run_eval.sh                # evaluate model
    make_plots.sh              # generate visualizations
    quick_cycle.sh             # fast demo loop
```

## Core Components

### 1. Agent Runtime (Rust)

**Crate: `agent`**

Minimal Pi-like loop with tools:
- `apply_patch(file, diff)` - Apply unified diff to file
- `run_tests()` - Execute cargo test in sandbox
- `run_check()` - Execute cargo check in sandbox
- `open_file(path)` - Read file contents
- `search(pattern)` - Search codebase

```rust
pub struct Agent {
    llm: Box<dyn LlmClient>,
    tools: ToolRegistry,
    policy: Policy,
    capture: EpisodeCapture,
}

impl Agent {
    pub async fn solve(&mut self, task: &Task) -> Episode {
        // 1. Present task + error to LLM
        // 2. LLM proposes tool calls
        // 3. Execute tools, capture results
        // 4. Loop until success or max attempts
        // 5. Return episode with full trace
    }
}
```

### 2. Episode Capture (Rust)

**Crate: `capture`**

```rust
pub struct Episode {
    pub task_id: String,
    pub attempt_idx: u32,
    pub prompt_hash: String,
    pub model_id: String,
    pub error_signature: String,      // normalized
    pub diff_unified: String,
    pub passed: bool,
    pub steps_to_green: u32,
    pub wall_clock_ms: u64,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub timestamp: DateTime<Utc>,
}
```

Storage: SQLite with JSONL export capability.

### 3. Sandbox (Rust)

**Crate: `sandbox`**

Runs cargo check/test in isolated temp directories:

```rust
pub struct Sandbox {
    temp_dir: TempDir,
    task: Task,
}

impl Sandbox {
    pub fn apply_patch(&mut self, diff: &str) -> Result<()>;
    pub fn run_check(&self) -> CompileResult;
    pub fn run_test(&self) -> TestResult;
}
```

### 4. Task System (Rust)

**Crate: `tasks_rust_koans`**

JSON task format:

```json
{
  "id": "task_0001",
  "family": "borrow_checker",
  "description": "Fix the moved value error",
  "buggy_code": "fn main() { let s = String::from(\"hello\"); let t = s; println!(\"{}\", s); }",
  "correct_code": "fn main() { let s = String::from(\"hello\"); let t = s.clone(); println!(\"{}\", s); }",
  "expected_error": "borrow of moved value"
}
```

Error families:
- `borrow_checker` - moved values, borrows
- `lifetimes` - lifetime annotations
- `trait_bounds` - missing trait implementations
- `result_handling` - Result/Option/? misuse
- `type_mismatch` - iterator types, generics

### 5. Training Pipeline (Python)

**Module: `sleepy_pact.train`**

```python
# lora_train.py
def train_lora(
    episodes_path: Path,
    base_model: str,
    output_dir: Path,
    max_steps: int = 500,
):
    # 1. Load episodes from SQLite
    # 2. Build training pairs (error+code -> fix)
    # 3. Fine-tune LoRA adapter
    # 4. Save adapter + metadata
```

### 6. Consolidation (Phase 2)

**Module: `sleepy_pact.train.merge_share`**

Share-style consolidation:

```python
def consolidate_adapters(
    adapters: List[Path],
    output_dir: Path,
    rank: int = 16,
):
    # 1. Load LoRA A,B matrices from all adapters
    # 2. Stack delta_W vectors
    # 3. Run SVD to find principal directions
    # 4. Keep top-k as shared basis B
    # 5. Compute coefficients c per task
    # 6. Save B and per-task c
```

**Module: `sleepy_pact.train.uwsh_basis`**

UWSH-style updates:

```python
def update_coefficients_only(
    basis: Path,
    new_episodes: Path,
    output_dir: Path,
):
    # 1. Load frozen basis B
    # 2. Train only coefficients c
    # 3. Much cheaper than full LoRA
```

### 7. Evaluation (Rust)

**Crate: `eval`**

```rust
pub struct EvalResult {
    pub repeat_error_rate: f64,    // same error signature seen before
    pub median_steps_to_green: f64,
    pub pass_rate: f64,
    pub regression_check: bool,    // frozen suite still passes
}

pub fn run_eval(
    agent: &mut Agent,
    evalset: &[Task],
    frozen_suite: &[Task],
) -> EvalResult;
```

### 8. Visualization (Python)

**Module: `sleepy_pact.viz`**

```python
def plot_learning_curve(
    runs_dir: Path,
    output_path: Path,
):
    # Plot repeat-error rate, steps-to-green, pass rate over iterations
```

## Data Flow

### Day Loop

```
Tasks -> Agent -> Episodes -> SQLite
                     |
                     v
              Error signatures
              (normalized)
```

### Sleep Loop

```
SQLite -> Curate -> SFT Dataset -> LoRA Train -> Adapter
                                       |
                                       v
                              (Phase 2: Consolidate)
                                       |
                                       v
                               Shared Basis + Coefficients
```

### Eval Loop

```
Frozen Suite + Adapter -> Agent -> Metrics
                                     |
                                     v
                              Gate: pass/fail
                                     |
                        +-----------+-----------+
                        |                       |
                        v                       v
                   Promote              Rollback
```

## LLM Integration

Supported inference backends:
- **Ollama** (default, local)
- **vLLM** (local, high performance)
- **HuggingFace TGI** (local or cloud)
- **OpenAI-compatible API** (generic)

Configuration via environment or config.yaml:

```yaml
llm:
  backend: ollama
  model: qwen2.5-coder:1.5b-instruct-q4_K_M
  base_url: http://localhost:11434
  max_tokens: 2048
```

## Key Design Decisions

1. **Rust for determinism**: Agent runtime, capture, and eval are in Rust for reproducibility
2. **Python for ML**: Training and consolidation use HuggingFace/PyTorch ecosystem
3. **SQLite for episodes**: Simple, portable, queryable
4. **Normalized error signatures**: Enable clustering and repeat detection
5. **Frozen eval sets**: Prevent regression, enable comparison across runs
6. **Gated promotions**: Never deploy a regressing adapter
