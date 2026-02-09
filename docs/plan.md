# Implementation Plan: Sleepy Coder

## Phase Overview

| Phase | Focus | Deliverables | Estimated Effort |
|-------|-------|--------------|------------------|
| 0 | Setup | Repo structure, tooling, CI | Foundation |
| 1 | MVP | Working day-sleep-eval loop | Core functionality |
| 2 | PaCT | Continual learning methods | Research demos |
| 3 | Production | Real repo usage | Practical tool |

---

## Phase 0: Project Setup

### 0.1 Repository Structure
- [ ] Create Rust workspace with crate structure
- [ ] Create Python package structure
- [ ] Set up data/ and runs/ directories
- [ ] Add .gitignore for artifacts

### 0.2 Tooling
- [ ] Configure Cargo.toml workspace
- [ ] Configure pyproject.toml with dependencies
- [ ] Set up pre-commit hooks (clippy, fmt, ruff)
- [ ] Add scripts/ shell scripts (bootstrap, build, run)

### 0.3 Documentation
- [ ] README.md with project overview
- [ ] CLAUDE.md for AI agent context
- [ ] docs/learnings.md initialized

### 0.4 CI/CD (Optional)
- [ ] GitHub Actions for Rust tests
- [ ] GitHub Actions for Python tests
- [ ] Automated markdown-checker

---

## Phase 1: MVP - Working Loop

### 1.1 Core Types (Rust)
- [ ] Define Episode struct with serde
- [ ] Define Task struct with serde
- [ ] Define EvalResult struct
- [ ] Schema versioning for migrations

### 1.2 Episode Capture (Rust)
- [ ] SQLite database wrapper
- [ ] Insert episode function
- [ ] Query episodes by run_id
- [ ] Export to JSONL

### 1.3 Sandbox (Rust)
- [ ] Create temp directory with task code
- [ ] Apply unified diff patches
- [ ] Run cargo check and capture output
- [ ] Run cargo test and capture output
- [ ] Parse and normalize error messages

### 1.4 Rust Koans Tasks
- [ ] Define task JSON format
- [ ] Create 10 borrow checker tasks
- [ ] Create 10 trait bounds tasks
- [ ] Create 10 Result/Option tasks
- [ ] Task loader from data/koans/

### 1.5 LLM Client (Rust)
- [ ] Ollama client implementation
- [ ] OpenAI-compatible API client
- [ ] Request/response types
- [ ] Token counting (approximate)

### 1.6 Agent Loop (Rust)
- [ ] Tool registry with apply_patch, run_check, run_test
- [ ] Minimal prompt template
- [ ] Parse LLM responses for tool calls
- [ ] Execute tools and collect results
- [ ] Loop until success or max attempts
- [ ] Capture full episode

### 1.7 CLI - Day Command (Rust)
- [ ] `sleepy-coder run-day --n 10`
- [ ] Load tasks from koans
- [ ] Run agent on each task
- [ ] Store episodes to SQLite
- [ ] Progress output

### 1.8 Episode Curator (Python)
- [ ] Load episodes from SQLite
- [ ] Filter failed-then-fixed episodes
- [ ] Build SFT dataset (input/output pairs)
- [ ] Save as JSONL in sft/

### 1.9 LoRA Trainer (Python)
- [ ] Load base model (HuggingFace)
- [ ] Configure PEFT LoraConfig
- [ ] Load SFT dataset
- [ ] Train with TrainingArguments
- [ ] Save adapter to sleep/adapter/

### 1.10 CLI - Sleep Command
- [ ] `sleepy-coder sleep`
- [ ] Call Python trainer
- [ ] Log training progress
- [ ] Save metadata (model, steps, loss)

### 1.11 Evaluation (Rust)
- [ ] Load frozen evaluation set
- [ ] Run agent with adapter applied
- [ ] Compute metrics: pass rate, steps-to-green
- [ ] Compute repeat-error rate
- [ ] Check gates
- [ ] Output metrics.jsonl

### 1.12 CLI - Eval Command
- [ ] `sleepy-coder eval`
- [ ] Load previous metrics for comparison
- [ ] Run evaluation
- [ ] Output gate pass/fail

### 1.13 Visualization (Python)
- [ ] Load metrics.jsonl from all cycles
- [ ] Plot repeat-error rate vs cycle
- [ ] Plot steps-to-green vs cycle
- [ ] Plot pass rate vs cycle
- [ ] Save PNGs to viz/

### 1.14 Quick Cycle Script
- [ ] scripts/quick_cycle.sh
- [ ] Run 6-12 cycles in sequence
- [ ] Generate final comparison plots

### 1.15 MVP Validation
- [ ] Run full cycle on 30 koans
- [ ] Verify metrics improve over 3+ cycles
- [ ] Verify no regression on frozen set
- [ ] Generate demo plots

---

## Phase 2: PaCT Methods (REVISED)

**Status**: In Progress - See [course-correction.md](./course-correction.md)

### 2.1 Generate Distinct Task Families (50 adapters)
- [ ] BorrowChecker variants (10 adapters): move, ref, mut, lifetime, copy/clone, rc/arc, cell, closure, async
- [ ] TraitSystem variants (10 adapters): derive, impl, bounds, where, associated, dyn, send/sync, from/into, iterator, display
- [ ] ErrorHandling variants (10 adapters): option, result, ?, match, combinators, custom error, thiserror, anyhow, unwrap, early return
- [ ] Rust2024 variants (10 adapters): fmt, let-chain, let-else, is_some_and, is_ok_and, copied, flatten, matches, clippy, async closure
- [ ] Advanced patterns (10 adapters): builder, newtype, typestate, phantom, unsafe, macro, proc_macro, simd, ffi, pin

### 2.2 Implement Share Algorithm (arXiv:2602.06043)
- [ ] `scripts/share_consolidate.py` - SVD-based subspace extraction
- [ ] Extract k principal basis vectors (60% explained variance)
- [ ] Compute per-adapter coefficients
- [ ] Save frozen basis + coefficient vectors
- [ ] Test on 10+ adapters

### 2.3 Coefficient-Only Training
- [ ] `scripts/train_coefficients.py` - Train only coefficients (basis frozen)
- [ ] Compare training cost vs full LoRA
- [ ] Validate no forgetting on frozen eval set
- [ ] Achieve pass rate ≥ 76.7% (baseline)

### 2.4 Incremental Subspace Update
- [ ] Implement merge algorithm (Algorithm 1, Phase 3)
- [ ] Reconstruct prior adapters from basis + coefficients
- [ ] Re-run SVD to update subspace
- [ ] Test with 20+ adapters arriving incrementally

### 2.5 Validation & Demos
- [ ] Compare: baseline vs naive LoRA vs Share
- [ ] Generate side-by-side comparison plots
- [ ] Document compression ratio (adapters → coefficients)
- [ ] Record demo videos

### Key Hyperparameters (from paper)
- **k**: Principal factors at 60% explained variance
- **p=1**: Pseudo-rank (higher values minimal benefit)
- **φ=[1, k/4]**: Temporary factors range

---

## Phase 3: Production Agent

### 3.1 Real Repo Support
- [ ] Scanner for real Rust projects
- [ ] Error extraction from cargo build output
- [ ] Diff capture from git

### 3.2 Per-Repo Coefficients
- [ ] Maintain coefficient sets per repo
- [ ] Switch coefficients on repo context
- [ ] Test personalization

### 3.3 Workflow Integration
- [ ] VSCode extension or CLI integration
- [ ] Background sleep training
- [ ] Notification on improvement

### 3.4 Scaling
- [ ] Support larger models (7B-13B)
- [ ] Support cloud GPU training
- [ ] Parallel evaluation

---

## Dependencies

### Rust Crates
```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rusqlite = { version = "0.31", features = ["bundled"] }
tempfile = "3"
reqwest = { version = "0.12", features = ["json"] }
clap = { version = "4", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
```

### Python Packages
```toml
[project]
dependencies = [
    "torch>=2.0",
    "transformers>=4.35",
    "peft>=0.7",
    "datasets>=2.14",
    "accelerate>=0.24",
    "matplotlib>=3.7",
    "pandas>=2.0",
    "numpy>=1.24",
]
```

---

## Milestones

1. **M1: Agent runs single task** - Sandbox works, LLM called, episode captured ✓
2. **M2: Day loop complete** - Run 30 tasks, store all episodes ✓
3. **M3: Sleep loop complete** - Train LoRA from episodes ✓
4. **M4: Eval loop complete** - Metrics computed, gates checked ✓
5. **M5: Full cycle works** - Day-sleep-eval-plot automated ✓
6. **M5a: Fix catastrophic forgetting** - Implement replay buffer, conservative training ✓
7. **M6: Learning demonstrated** - Metrics improve over 3+ cycles (**BLOCKED - needs Share**)
8. **M7: Share consolidation works** - Multiple adapters merged (**IN PROGRESS**)
9. **M8: Demo videos complete** - YouTube content published

## Current Status (2026-02-09)

### Completed
- Phase 0: Setup ✓
- Phase 1: MVP (72 tests passing) ✓
- Baseline evaluation: 76.7% pass rate ✓
- CUDA training pipeline working ✓
- GGUF quantization and Ollama export ✓
- Gate check script ✓
- 12 training cycles run ✓

### Critical Finding
**Share algorithm not properly implemented.** After 12 cycles:
- Best result: 73.3% (3.4% below baseline)
- We used replay + merged adapters (NOT Share)
- Share requires: SVD basis extraction + coefficient-only training

See `docs/course-correction.md` for full analysis.

### Next: Proper Share Implementation
1. Generate 50 distinct task families
2. Train adapter per family
3. SVD-based subspace extraction
4. Coefficient-only training for new tasks

---

## Risk Mitigation

| Risk | Mitigation | Status |
|------|------------|--------|
| Model too good (no errors to learn from) | Use deliberately weak model, harder tasks | ✓ Works |
| Training unstable | Smaller learning rate, gradient clipping, validation | ✓ Works |
| Regression creep | Strict gates, frozen eval set, automatic rollback | ✓ Implemented |
| Slow iteration | Micro-cycles (10 tasks, 300 steps), parallel eval | ✓ Works |
| VRAM constraints | 4-bit quantization, gradient checkpointing, small batch | ✓ Works |
| **Catastrophic forgetting** | **Replay buffer (50%+), mixed data, lower LR** | **✓ Fixed** |

## Path to Better Results

### Immediate Next Steps (30-60 min)

```bash
# 1. Retrain with proper approach
cd cuda
source .venv/bin/activate
python scripts/train.py --steps 100 --lr 1e-4

# 2. Export and evaluate
python scripts/merge.py --adapter ../runs/adapters/<latest>/adapter
# Then: ollama create, evaluate

# 3. Compare to baseline
python ../scripts/update_dashboard.py
```

### Expected Timeline

| Phase | Duration | Expected Outcome |
|-------|----------|------------------|
| Single corrected cycle | 30-60 min | Pass rate ≥ 76.7% (no regression) |
| 3-cycle validation | 2-3 hours | Upward trend visible |
| 5-cycle full demo | 4-5 hours | Clear learning curve for demo |
| Video recording | 1-2 hours | 2-min short + 15-min explainer |

### Success Criteria for Demo

1. **Learning curve plot** showing improvement over 3+ cycles
2. **No regression** below baseline on any cycle
3. **Pass rate improvement** of at least +5% after 5 cycles
4. **Dashboard** with real-time metrics
