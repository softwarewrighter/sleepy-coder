# CUDA Next Steps (Arch/RTX 5060)

This document provides context for Claude CLI to continue Share algorithm experiments on the CUDA system.

---

## Context

We've made progress on Share paper validation:

1. **Routing tested (Exp 1a/1b)** - Gradient-trained routing shows zero regressions
2. **Coefficient training works** - v4 coefficients properly trained with both-random init
3. **Key finding**: Routing protects unrelated koans (rh_008 preserved)

Current results:
- Prompt engineering: 83.3%
- Share averaged: 50.0%
- Share routed (v4): 50.0% (but zero regressions, RH +10%)

See `docs/paper-checklists.md` for the full checklist.
See `docs/status.md` for Experiment 1a/1b details.

---

## Priority Experiments (in order)

### Experiment 1: DONE - Routing vs Averaging

**Status**: Tested in Exp 1a (analytical) and Exp 1b (gradient-trained v4)

**Results**:
- Analytical projection: 43.3% (hurt)
- Gradient-trained v4: 50.0% routed, 50.0% averaged
- **Key win**: Routing prevents rh_008 regression that averaging causes

**Next improvements**:
1. Fix k_alpha=32 (currently 174, paper recommends ~32)
2. Add rank update vectors
3. Results saved in `runs/experiments/forgetting/`

### Experiment 2: Sequential Learning Curve (NOT DONE)

**Goal**: Demonstrate continual learning (train task 1, then 2, then 3...).

**Steps**:
1. Start with base model (no adapters)
2. Train coefficient for task 1 → eval ALL 30 tasks
3. Train coefficient for task 2 → eval ALL 30 tasks (verify task 1 still passes)
4. Repeat for 5-10 tasks
5. Plot learning curve showing no forgetting

**Expected**: Flat curve on previously-learned tasks (no degradation).

**This is the KEY experiment** to demonstrate the paper's core claim about preventing catastrophic forgetting.

### Experiment 3: UWSH Subspace Stability (NOT DONE)

**Goal**: Verify universal subspace hypothesis.

**Steps**:
1. Split 51 adapters into two subsets (A and B)
2. Extract subspace from subset A
3. Extract subspace from subset B
4. Compute Grassmann distance between subspaces
5. **Target**: >70% overlap validates UWSH

---

## Environment Setup (Arch/CUDA)

```bash
# Activate environment
cd ~/github/sleepy-coder
source .venv/bin/activate  # or conda activate sleepy

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verify existing scripts work
python scripts/share_complete.py --help
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/share_complete.py` | Main Share algorithm (all 3 phases) |
| `scripts/routed_inference.py` | Error classification + routing |
| `scripts/generate_targeted_v2.py` | Training data generation |
| `runs/share_proper_trained/` | Saved Share basis + 75 coefficients |
| `docs/paper-checklists.md` | Full experiment checklist |
| `docs/share-paper-abstract-and-notes.md` | Paper notes |

## The 7 Failing Patterns

These consistently fail and are the target for routing/training:

| Task | Error Pattern |
|------|--------------|
| bc_003 | immutable/mutable borrow conflict |
| bc_005 | double mutable borrow |
| bc_010 | return local reference |
| tb_002 | missing Clone |
| tb_007 | missing Hash |
| tb_008 | missing Ord |
| rh_004 | Option to Result |

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Pass rate (routed) | 73.3% (avg) | ≥ 80% |
| Forgetting rate | ~10% (full LoRA) | 0% |
| Sequential learning | Not tested | No degradation |

---

## Quick Start Commands

```bash
# Pull latest
git pull

# Check existing Share artifacts
ls runs/share_proper_trained/

# Run eval with current model (baseline)
cd rust && cargo run -- eval --cycle 100

# Test routing implementation
python scripts/routed_inference.py --test
```

---

## What Claude Should Do

1. Review `docs/paper-checklists.md` for context
2. Implement routing in `scripts/routed_inference.py` if not complete
3. Run Experiment 1 (routing vs averaging)
4. If routing helps, proceed to Experiment 2 (coefficient-only training)
5. Document results in `docs/status.md`
