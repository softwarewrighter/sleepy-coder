# Project Status: Sleepy Coder

## Current Phase: Phase 2 - PaCT (Course Correction)

**Last Updated**: 2026-02-09

---

## Overall Progress

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Setup | Complete | 100% |
| Phase 1: MVP | Complete | 100% |
| Phase 2: PaCT | **In Progress** | 20% |
| Phase 3: Production | Not Started | 0% |

## Critical Finding: Share Not Properly Implemented

After 12 training cycles, best result was 73.3% (3.4% below baseline).
**Root cause**: We did not implement the Share paper correctly.

See [docs/course-correction.md](./course-correction.md) for full analysis.

---

## Phase 1: MVP

### Completed
- [x] Phase 1.1: Core Types (core_types) - 12 tests
- [x] Phase 1.2: Episode Capture (capture) - 9 tests
- [x] Phase 1.3: Sandbox (sandbox) - 9 tests
- [x] Phase 1.4: Rust Koans Tasks (tasks_rust_koans) - 10 tests, 42 builtin koans
- [x] Phase 1.5: LLM Client (agent crate) - OllamaClient with generate/generate_fix
- [x] Phase 1.6: Agent Loop (agent crate) - RED-patch-GREEN loop, 13 tests total
- [x] Phase 1.7: Evaluation harness (eval crate) - 12 tests, metrics and run comparison
- [x] Phase 1.8: CLI (cli crate) - 7 tests, run/eval/list/show commands
- [x] Helper scripts (build-all, test-all, lint, quick-cycle)

### Documentation
- [x] README.md with project overview
- [x] CLAUDE.md for AI agent context
- [ ] docs/learnings.md initialized

---

## Phase 0: Project Setup (Complete)

- [x] Initial repository created
- [x] docs/research.txt - Captured research conversation
- [x] docs/ai_agent_instructions.md - AI agent guidelines
- [x] docs/process.md - Development process
- [x] docs/tools.md - Tool documentation
- [x] docs/prd.md - Product requirements
- [x] docs/architecture.md - System architecture
- [x] docs/design.md - Technical design
- [x] docs/plan.md - Implementation plan
- [x] docs/status.md - This file
- [x] docs/references.md - Paper links
- [x] Create Rust workspace structure
- [x] Configure Cargo.toml workspace
- [x] Create Python package structure
- [x] Configure pyproject.toml
- [x] Set up data/ and runs/ directories
- [x] .gitignore configured

---

## Blockers

None currently.

---

## Recent Activity

### 2026-02-09
- Ran 12 training cycles with various configurations
- Best result: 73.3% pass rate (cycles 9, 10, 12)
- **Critical discovery**: Share algorithm not properly implemented
- Created course-correction.md with proper implementation plan
- Generated Rust 2024 koans (format strings, let-chains, modern methods)
- Created gate_check.py script for deployment safety
- Created generate_koans_large.py for expanded training data
- **Implemented proper Share algorithm** (share_proper.py):
  - Keeps B and A matrices separate (key fix)
  - SVD on stacked matrices independently
  - Extracts k principal components per layer
  - Stores 1.58M params shared basis + 10K coefficients per task
- Trained 6 diverse domain adapters (yew_wasm, axum_server, sqlx_db, cli_clap, refactoring, style_metrics)
- Consolidated with Share: k=6, p=2, MSE=0 (perfect reconstruction)
- **C13 result: 73.3%** - Same as previous ceiling, training data doesn't overlap with eval koans

### Training Cycle Results
| Cycle | Approach | Pass Rate | Notes |
|-------|----------|-----------|-------|
| C0 | Baseline | 76.7% | Target to beat |
| C1 | Naive LoRA | 60.0% | Catastrophic forgetting |
| C9-10 | Minimal (20 steps) | 73.3% | Best fine-tuned |
| C12 | Rust 2024 | 73.3% | Same ceiling |
| C13 | **Share (proper)** | 73.3% | 6 domains consolidated, k=6, p=2 |

### 2026-02-08
- Created initial documentation structure
- Captured research from ChatGPT conversation
- Generated PRD, architecture, design, plan, and status docs
- Implemented core_types crate (Episode, Task, EvalResult) - 12 tests
- Implemented capture crate (EpisodeStore with SQLite) - 9 tests
- Implemented sandbox crate (isolated cargo check/test) - 9 tests
- Added helper scripts (build-all, test-all, lint, quick-cycle)
- Implemented tasks_rust_koans crate - 10 tests, 42 builtin koans across 5 error families
- Implemented agent crate with OllamaClient and AgentLoop - 13 tests
- Implemented eval crate with EvalHarness, EvalMetrics, run comparison - 12 tests
- Implemented CLI with run/eval/list/show commands - 7 tests
- **Phase 1: MVP Complete** - 72 tests total
- CUDA training pipeline working
- GGUF quantization and Ollama export working

---

## Next Steps

1. **Implement Share Algorithm Properly** - See course-correction.md
2. **Generate 50 distinct task families** - Truly novel training data
3. **SVD-based subspace extraction** - Principal basis + frozen coefficients
4. **Coefficient-only training** - For new tasks (cheaper than full LoRA)

---

## Metrics

| Metric | Baseline (C0) | Best (C9-12) | Target |
|--------|---------------|--------------|--------|
| Pass Rate | **76.7%** | 73.3% | ≥ 76.7% |
| Median Steps to Green | 2.0 | 2.0 | ≤ 2.0 |
| Error Families |  |  |  |
| - BorrowChecker | 70.0% | 70.0% | ≥ 70% |
| - ResultHandling | 90.0% | 70-90% | ≥ 90% |
| - TraitBounds | 70.0% | 60-70% | ≥ 70% |

---

## Links

- Research Papers:
  - [Share LoRA Subspaces](https://arxiv.org/html/2602.06043v1)
  - [UWSH](https://arxiv.org/abs/2512.05117)
- Related Projects:
  - Pi minimal agent (OpenClaw/ClaudBot)
  - opencode
