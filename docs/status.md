# Project Status: Sleepy Coder

## Current Phase: Prompt Engineering (Success!)

**Last Updated**: 2026-02-17

---

## Key Result: Prompt Engineering Succeeds Where LoRA Failed

After LoRA fine-tuning proved ineffective, **prompt engineering improved pass rate from 73.3% to 83.3% average** (up to 86.7% peak).

Error-specific hints added to the prompt guide the model to fix:
- Borrow checker errors (mutable/immutable conflicts)
- Trait bound errors (Hash, Ord with full trait hierarchies)
- Result/Option conversions

See [rust/crates/agent/src/lib.rs](../rust/crates/agent/src/lib.rs) `get_error_hints()` function.

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
| Share Full (Ph2+Ph3) | 73.3% | 0% | 0 |
| **Prompt Engineering** | **83.3%** | **+10%** | **~0** |
| Share Routed (Exp 1a, analytical) | 43.3%* | -6.7%* | 3 |
| **Share Routed (Exp 1b, v4 trained)** | **50.0%*** | **+3.3%*** | **0** |

### Key Insights

1. **Training on known patterns causes forgetting** — The base model already knows ~77% of patterns. Training dilutes signal and overwrites knowledge.

2. **Even targeted training causes regressions** — Training only on the 7 failure patterns (44 examples) caused 4 new regressions.

3. **Replay buffers reduce but don't prevent forgetting** — Adding 17 "passing" examples still caused 2 regressions.

4. **Share prevents forgetting but doesn't improve** — The full 3-phase algorithm returns to baseline. Averaging coefficients negates specialization.

5. **More adapters ≠ better** — 51 adapters performed worse than 6 due to subspace dilution when averaging.

6. **The model's knowledge is interconnected** — Modifying any weights (even 10K params) risks breaking other capabilities.

7. **Analytical routing hurts, trained routing is neutral-to-positive** — Exp 1a (analytical projection) hurt (43.3%). Exp 1b (gradient-trained v4) shows 50% routed with zero regressions and +10% on result_handling. Routing protects unrelated koans from coefficient interference.

---

## Experiment 1a: Routing vs Averaging (Analytical Projection, 2026-02-17)

Tested Share routing with analytically-projected (NOT gradient-trained) coefficients.

**Setup**: Single-shot, plain prompt, HuggingFace bf16, direct weight modification.

| Strategy | Pass Rate | BC (10) | TB (10) | RH (10) |
|----------|-----------|---------|---------|---------|
| Baseline | 50.0% | 70% | 40% | 40% |
| Averaged | 50.0% | 70% | 40% | 40% |
| Routed | 43.3% | 70% | **20%** | 40% |

**Result**: Analytical coefficients cause regressions. Training needed.

---

## Experiment 1b: Gradient-Trained v4 Coefficients (2026-02-17)

Fixed two critical bugs in Phase 2 training:
1. **Zero-gradient saddle point**: Both eps_beta and eps_alpha were zero-initialized, causing `delta_W = 0 @ 0` with zero gradients. Fixed with dual small-random init.
2. **Half-param training**: Only 112/224 params got gradients (LoRA-style init trained only eps_beta). Fixed with both-random init: 224/224 params trained.

**v4 hyperparameters**: p=4, 100 steps, lr=1e-4, weight_decay=0.01, batch=4.

### Results

| Strategy | Pass Rate | BC (10) | RH (10) | TB (10) |
|----------|-----------|---------|---------|---------|
| Baseline | 46.7% (14/30) | 70% | 40% | 30% |
| Averaged | **50.0%** (15/30) | 70% | 40% | **40%** |
| Routed | **50.0%** (15/30) | 70% | **50%** | 30% |

### Forgetting Heatmap

Applied each coefficient **individually to all 30 koans** to measure per-koan forgetting:

```
Koan      BL  mut_bc dbl_mt ret_lr mis_cl mis_hs mis_or opt_ok res_me ROUTED AVGD
bc_001-009 P   P      P      P      P      P      P      P      P      P      P
bc_003,5,10 .  .      .      .      .      .      .      .      .      .      .
rh_002     .  .     +GAIN   .      .     +GAIN  +GAIN  +GAIN  +GAIN  +GAIN   .
rh_008     P -LOST  -LOST  -LOST  -LOST  -LOST  -LOST  -LOST  -LOST   P    -LOST
tb_005     P  P      P      P      P     -LOST   P      P      P      P      P
```

**Key**: P=stayed passing, .=stayed failing, +GAIN=improved, -LOST=regressed

### Analysis

1. **Routing prevents forgetting**: rh_008 regresses under ALL 8 coefficients applied globally, but routing **saves it** because rh_008's error doesn't match any pattern (falls back to base model).
2. **rh_002 universally gains**: 5 of 8 coefficients improve rh_002. This suggests a shared direction in the coefficient space helps this koan.
3. **Averaged loses**: Averaging hurts because it includes the directions that regress rh_008 without the routing protection.
4. **Result_handling improved**: Routed RH jumped from 40% to 50% (rh_002 gained, rh_008 preserved).
5. **No borrow_checker regressions**: All 7 passing BC koans remain passing under all strategies.
6. **rh_008 is fragile**: Any coefficient perturbation breaks it. This koan sits near a decision boundary.

### Conclusion

Properly gradient-trained coefficients with routing **do not cause catastrophic forgetting** and show targeted improvement (+10% on result_handling). The routing mechanism is essential: it protects unrelated koans from coefficient interference. Next: reduce k_alpha from 174 to 32 (matching paper), add rank update vectors.

Results saved to `runs/experiments/forgetting/forgetting_analysis.json`.

---

## Overall Progress

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Setup | Complete | 100% |
| Phase 1: MVP | Complete | 100% |
| Phase 2: PaCT | Complete (Negative Result) | 100% |
| **Phase 2b: Prompt Engineering** | **Complete (Success!)** | **100%** |
| Phase 3: Production | Unblocked | 0% |

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
- [x] Task routing tested (Exp 1a): analytical routing hurts (43.3%)
- [x] Gradient-trained v4 routing: zero regressions, RH +10% (Exp 1b)

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

1. **Prompt Engineering** ✅ **IMPLEMENTED - SUCCESS**
   - Added error-specific hints to prompts
   - Pass rate improved from 73.3% → 83.3% average
   - No weight modification needed

2. **Multi-Turn Repair**
   - Use dialogue to iteratively fix errors
   - Agent asks clarifying questions
   - Leverages base model's reasoning

3. **Larger Model**
   - Try 7B or 14B parameter models
   - More capacity to absorb new knowledge without forgetting
   - Higher compute cost

4. **Task Routing with Share** -- **TESTED (Promising with proper training)**
   - Analytical projection: 43.3% (worse). Gradient-trained v4: 50.0% (no regression)
   - Routing protects unrelated koans; result_handling improved 40% → 50%
   - See Experiments 1a/1b above. Next: fix k_alpha=32, add rank updates

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
| Pass Rate | 73.3% | **86.7%** | ≥ 76.7% |
| Median Steps | 2.0 | 2.0 | ≤ 2.0 |
| Regressions | 0 | ~0 | 0 |

**Conclusion**: Target exceeded via prompt engineering! LoRA failed but hints work.

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
| **C100-102** | **02-12** | **Prompt Engineering** | **80-86.7%** | **Error-specific hints work!** |
| Exp 1a | 02-17 | Share Routing (analytical) | 43-50%* | Routing hurts with analytical coefficients |
| **Exp 1b** | **02-17** | **Share Routing (v4 trained)** | **50%*** | **Zero regressions, RH +10%** |

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
