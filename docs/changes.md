# Sleepy-Coder: Issues, Corrections, and New Approach

*Document created: 2026-02-08*

## Executive Summary

Our initial naive approach to continual learning caused **catastrophic forgetting**, with the model regressing from 76.7% to 60.0% pass rate (-16.7%). This document captures what went wrong, what the research says, and the corrected approach.

---

## Issue #1: Catastrophic Forgetting

### What Happened
- Trained on 23 failed episodes using pure supervised fine-tuning (SFT)
- Model performance **decreased** by 16.7% instead of improving
- The trained model forgot general Rust coding patterns while overfitting to specific error corrections

### Root Cause Analysis

| Factor | Our Implementation | Research Recommendation |
|--------|-------------------|------------------------|
| Data volume | 23 examples | Hundreds to thousands |
| Data composition | Failed examples only | Mix of success + failure + replay |
| Replay buffer | None | 50%+ from original training |
| Learning rate | 2e-4 | 1e-4 or lower |
| LoRA rank | r=16 | r=8 (more conservative) |

### Research Evidence

1. **[ACL 2024 - Self-Synthesized Rehearsal](https://aclanthology.org/2024.acl-long.77/)**: LLMs need synthetic data generation for rehearsal to prevent forgetting
2. **[SURE 2024](https://openreview.net/pdf?id=IgZWU75BLL)**: Replay consistently outperforms regularization methods like EWC
3. **[EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.249/)**: Direct link between loss landscape flatness and catastrophic forgetting
4. **[RL vs SFT 2024](https://arxiv.org/html/2508.16546v1)**: RL methods generalize better; SFT tends to memorize

---

## Issue #2: Naive Training Pipeline

### What Was Wrong

```
Old Pipeline:
  failed_episodes → SFT → export → evaluate → REGRESSION
```

Problems:
1. **No baseline preservation**: Training overwrote existing capabilities
2. **Single-shot training**: No iterative refinement with feedback
3. **No quality gating**: Deployed model without regression check
4. **Wrong hyperparameters**: LR too high, LoRA rank too high

### Corrected Pipeline

```
New Pipeline:
  ┌──────────────────────────────────────────────────────────┐
  │  1. PREPARE DATA                                         │
  │     ├── 44% replay (original SFT examples)               │
  │     ├── 42% success reinforcement (passed tasks)         │
  │     └── 14% hard examples (failed tasks, extra weight)   │
  ├──────────────────────────────────────────────────────────┤
  │  2. TRAIN (conservative)                                 │
  │     ├── lr = 1e-4 (halved from before)                   │
  │     ├── LoRA r=8 (reduced from 16)                       │
  │     ├── Steps per cycle = 100 (not 500)                  │
  │     └── LR decay = 10% per cycle                         │
  ├──────────────────────────────────────────────────────────┤
  │  3. EVALUATE                                             │
  │     ├── Run on frozen eval set                           │
  │     ├── Compare to baseline                              │
  │     └── Gate: only deploy if improved                    │
  ├──────────────────────────────────────────────────────────┤
  │  4. ITERATE                                              │
  │     └── Repeat for N cycles with updated data            │
  └──────────────────────────────────────────────────────────┘
```

---

## Issue #3: Missing Visualization and Documentation

### What Was Wrong
- No real-time tracking of learning progress
- No visualization of training curves
- No comparison plots between cycles
- Results not documented for reproducibility

### Corrections Made

Created visualization infrastructure:

| File | Purpose |
|------|---------|
| `docs/dashboard.html` | Interactive Chart.js dashboard |
| `docs/results.png` | Static comparison plot |
| `docs/results.md` | Documented analysis with research citations |
| `scripts/update_dashboard.py` | Auto-update dashboard from metrics |
| `scripts/generate_results.py` | Generate plots and documentation |

---

## New Scripts Created

### 1. `scripts/prepare_training_data.py`
Implements research-backed data preparation:
- Full replay of original training data
- Extra copies of hard examples (tasks that failed)
- Success reinforcement (duplicate passed tasks)
- Proper shuffling and mixing

### 2. `scripts/continual_train.py`
Multi-cycle training with:
- Configurable number of cycles
- Learning rate decay per cycle
- Evaluation after each cycle
- Learning curve generation
- Metrics history tracking

### 3. `scripts/update_dashboard.py`
Dashboard synchronization:
- Reads metrics from JSONL
- Updates Chart.js data
- Keeps visualization current

---

## Key Learnings

### 1. Replay is Essential
> "Replay consistently outperforms regularization-based methods such as EWC and O-LoRA across benchmarks." - SURE 2024

We now include 100% of original training data in every training run.

### 2. More Data Types, Not Just Failures
Training on failures alone teaches the model bad patterns. We now use:
- **Replay** (44%): Preserve base capabilities
- **Success** (42%): Reinforce what works
- **Hard cases** (14%): Focus on problems

### 3. Conservative Updates
Smaller changes = less forgetting:
- Lower learning rate (1e-4 vs 2e-4)
- Lower LoRA rank (r=8 vs r=16)
- More frequent checkpoints
- Shorter training cycles with evaluation

### 4. Gate Before Deploying
Never deploy a model without comparing to baseline. The eval harness must pass gates before promotion.

---

## Timeline to Show Better Results

### Immediate (Today, ~30 min)
1. Run `prepare_training_data.py` to create mixed dataset ✓
2. Retrain with new pipeline
3. Evaluate and compare

### Short-term (1-2 hours)
1. Run 3-5 training cycles with proper approach
2. Generate learning curve showing improvement
3. Update dashboard with new results

### Validation Criteria
A successful run should show:
- [ ] Pass rate ≥ baseline (76.7%)
- [ ] No regression vs previous cycle
- [ ] Learning curve trending upward

---

## Technical Debt Addressed

| Issue | Status |
|-------|--------|
| Package management docs (pip vs uv) | ✓ Fixed in CLAUDE.md |
| Missing visualization | ✓ Created dashboard.html |
| No replay buffer | ✓ Added to prepare_training_data.py |
| Single-cycle training | ✓ Created continual_train.py |
| Missing research citations | ✓ Added to results.md |

---

## References

1. [Mitigating Catastrophic Forgetting with Self-Synthesized Rehearsal (ACL 2024)](https://aclanthology.org/2024.acl-long.77/)
2. [SURE: Surprise-Driven Prioritised Replay (2024)](https://openreview.net/pdf?id=IgZWU75BLL)
3. [Revisiting Catastrophic Forgetting in LLM Tuning (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.249/)
4. [RL vs Supervised Fine-Tuning for LLMs (2024)](https://arxiv.org/html/2508.16546v1)
5. [Continual Learning of Large Language Models Survey (CSUR 2025)](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
