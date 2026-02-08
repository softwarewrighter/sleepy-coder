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

## Phase 2: PaCT Methods

### 2.1 Share-Style Consolidation
- [ ] Implement delta_W extraction from LoRA
- [ ] Implement SVD-based basis computation
- [ ] Implement coefficient projection
- [ ] Save shared basis + per-cycle coefficients
- [ ] Test consolidation of 3+ adapters

### 2.2 UWSH Coefficient Updates
- [ ] Implement frozen basis loading
- [ ] Implement coefficient-only training
- [ ] Compare training cost vs full LoRA
- [ ] Validate performance retention

### 2.3 Orthogonal Constraint (Optional)
- [ ] Implement O-LoRA style constraint
- [ ] Compare forgetting vs baseline

### 2.4 Method Comparison Demos
- [ ] Run same koans with: no training, LoRA, Share, UWSH
- [ ] Generate side-by-side comparison plots
- [ ] Document method trade-offs

### 2.5 Demo Videos
- [ ] Record 2-min Short: "This AI learns from mistakes"
- [ ] Record 15-min explainer with plots
- [ ] Blog post with repo link

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

1. **M1: Agent runs single task** - Sandbox works, LLM called, episode captured
2. **M2: Day loop complete** - Run 30 tasks, store all episodes
3. **M3: Sleep loop complete** - Train LoRA from episodes
4. **M4: Eval loop complete** - Metrics computed, gates checked
5. **M5: Full cycle works** - Day-sleep-eval-plot automated
6. **M6: Learning demonstrated** - Metrics improve over 6+ cycles
7. **M7: Share consolidation works** - Multiple adapters merged
8. **M8: Demo videos complete** - YouTube content published

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model too good (no errors to learn from) | Use deliberately weak model, harder tasks |
| Training unstable | Smaller learning rate, gradient clipping, validation |
| Regression creep | Strict gates, frozen eval set, automatic rollback |
| Slow iteration | Micro-cycles (10 tasks, 300 steps), parallel eval |
| VRAM constraints | 4-bit quantization, gradient checkpointing, small batch |
