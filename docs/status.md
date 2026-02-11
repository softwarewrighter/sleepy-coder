# Project Status: Sleepy Coder

## Current Phase: Phase 2 - PaCT (Complete - Negative Result)

**Last Updated**: 2026-02-11

---

## Critical Finding: LoRA Fine-Tuning Cannot Improve This Model

After extensive experimentation (51 adapters, multiple algorithms, various hyperparameters), we have conclusively determined that **LoRA fine-tuning cannot improve the base model (Qwen2.5-Coder-1.5B-Instruct) for this task**.

Every approach tried caused regressions. The baseline represents the ceiling.

See [docs/learnings.md](./learnings.md) for detailed experiment logs.

---

## Experiment Summary

| Approach | Pass Rate | vs Baseline | Regressions |
|----------|-----------|-------------|-------------|
| Baseline (no training) | 73.3% | — | 0 |
| Naive LoRA (C1) | 60.0% | -13.3% | Many |
| Share-51 averaged | 73.3% | 0% | 0 |
| Targeted full LoRA (7 patterns) | 63.3% | -10% | 4+ |
| Replay buffer | 70.0% | -3.3% | 2 |
| Phase 2 coef-only (10K params) | 66.7% | -6.6% | 2 |
| **Share Full (Ph2+Ph3)** | **73.3%** | **0%** | **0** |

### Key Insights

1. **Training on known patterns causes forgetting** — The base model already knows ~77% of patterns. Training dilutes signal and overwrites knowledge.

2. **Even targeted training causes regressions** — Training only on the 7 failure patterns (44 examples) caused 4 new regressions.

3. **Replay buffers reduce but don't prevent forgetting** — Adding 17 "passing" examples still caused 2 regressions.

4. **Share prevents forgetting but doesn't improve** — The full 3-phase algorithm returns to baseline. Averaging coefficients negates specialization.

5. **More adapters ≠ better** — 51 adapters performed worse than 6 due to subspace dilution when averaging.

6. **The model's knowledge is interconnected** — Modifying any weights (even 10K params) risks breaking other capabilities.

---

## Overall Progress

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Setup | Complete | 100% |
| Phase 1: MVP | Complete | 100% |
| Phase 2: PaCT | **Complete (Negative Result)** | 100% |
| Phase 3: Production | Blocked | 0% |

---

## Phase 2: PaCT Results

### What We Built
- [x] CUDA training pipeline (QLoRA, Flash Attention)
- [x] GGUF export and Ollama integration
- [x] Share algorithm (proper SVD-based implementation)
- [x] 51 pattern-specific LoRA adapters
- [x] Phase 2 coefficient-only training
- [x] Phase 3 basis merging
- [x] Extensive training data (431+ examples across 50+ patterns)

### What We Learned
- [x] LoRA fine-tuning causes catastrophic forgetting on capable base models
- [x] Share algorithm prevents forgetting but can't improve when averaging
- [x] The base model's 73-77% pass rate is likely the ceiling
- [x] Task routing (not averaging) may be needed for Share to help

---

## Phase 1: MVP (Complete)

- [x] Core Types (core_types) - 12 tests
- [x] Episode Capture (capture) - 9 tests
- [x] Sandbox (sandbox) - 9 tests
- [x] Rust Koans Tasks (tasks_rust_koans) - 10 tests, 42 builtin koans
- [x] LLM Client (agent crate) - OllamaClient
- [x] Agent Loop (agent crate) - RED-patch-GREEN loop, 13 tests
- [x] Evaluation harness (eval crate) - 12 tests
- [x] CLI (cli crate) - 7 tests

**Total: 72 tests passing**

---

## Possible Paths Forward

Since LoRA fine-tuning is not viable, alternative approaches:

1. **Prompt Engineering**
   - Improve system prompts with error-specific guidance
   - Add few-shot examples in context
   - No weight modification needed

2. **Multi-Turn Repair**
   - Use dialogue to iteratively fix errors
   - Agent asks clarifying questions
   - Leverages base model's reasoning

3. **Larger Model**
   - Try 7B or 14B parameter models
   - More capacity to absorb new knowledge without forgetting
   - Higher compute cost

4. **Task Routing with Share**
   - Don't average coefficients
   - Detect error type, select appropriate adapter
   - Requires error classification

5. **Model Ensemble**
   - Run multiple models, pick best output
   - Or use voting/consensus

6. **Accept Baseline**
   - 73-77% may be acceptable for this model+task
   - Focus on other improvements (UX, speed, reliability)

---

## Metrics

| Metric | Baseline | Best Achieved | Target |
|--------|----------|---------------|--------|
| Pass Rate | **73.3%** | 73.3% | ≥ 76.7% |
| Median Steps | 2.0 | 2.0 | ≤ 2.0 |
| Regressions | 0 | 0 | 0 |

**Conclusion**: Target not achievable with LoRA on this base model.

---

## Training Cycle History

| Cycle | Date | Approach | Pass Rate | Notes |
|-------|------|----------|-----------|-------|
| C0 | 02-08 | Baseline | 76.7% | Initial target |
| C1 | 02-08 | Naive LoRA | 60.0% | Catastrophic forgetting |
| C9-10 | 02-09 | Minimal (20 steps) | 73.3% | Best fine-tuned |
| C12 | 02-09 | Rust 2024 | 73.3% | Same ceiling |
| C13 | 02-09 | Share (6 adapters) | 73.3% | Proper implementation |
| C14+ | 02-10 | Share (51 adapters) | 70.0-73.3% | More adapters = worse |
| Final | 02-10 | Share Full (Ph2+Ph3) | 73.3% | Prevents forgetting, no improvement |

---

## Documentation

- [x] [README.md](../README.md) - Project overview with diagrams
- [x] [CLAUDE.md](../CLAUDE.md) - AI agent context
- [x] [learnings.md](./learnings.md) - Detailed experiment logs
- [x] [story.md](./story.md) - Project narrative
- [x] [next-steps.md](./next-steps.md) - Analysis and recommendations
- [x] [course-correction.md](./course-correction.md) - Share implementation details
- [x] [changes.md](./changes.md) - Training history

---

## Links

- Research Papers:
  - [Share LoRA Subspaces (arXiv:2602.06043)](https://arxiv.org/abs/2602.06043)
  - [UWSH (arXiv:2512.05117)](https://arxiv.org/abs/2512.05117)
- Project:
  - [GitHub Repository](https://github.com/softwarewrighter/sleepy-coder)
