# Project Status: Sleepy Coder

## Current Phase: Phase 1 - MVP

**Last Updated**: 2026-02-08

---

## Overall Progress

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Setup | Complete | 100% |
| Phase 1: MVP | Complete | 100% |
| Phase 2: PaCT | Not Started | 0% |
| Phase 3: Production | Not Started | 0% |

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
- [ ] README.md with project overview
- [ ] CLAUDE.md for AI agent context
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

---

## Next Steps

1. **README.md** - Project overview and getting started guide
2. **CLAUDE.md** - AI agent context file
3. **Integration testing** - End-to-end tests with Ollama running
4. **Phase 2: PaCT** - Implement LoRA training pipeline

---

## Metrics

(Will be populated once MVP is running)

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Repeat Error Rate | - | - | < 20% after 6 cycles |
| Median Steps to Green | - | - | < 3.0 after 6 cycles |
| Frozen Set Pass Rate | - | - | > 90% maintained |

---

## Links

- Research Papers:
  - [Share LoRA Subspaces](https://arxiv.org/html/2602.06043v1)
  - [UWSH](https://arxiv.org/abs/2512.05117)
- Related Projects:
  - Pi minimal agent (OpenClaw/ClaudBot)
  - opencode
