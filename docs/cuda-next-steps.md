# CUDA Next Steps (Arch/RTX 5060)

This document provides context for Claude CLI to continue Share algorithm experiments on the CUDA system.

---

## Context

We identified critical gaps between our Share implementation and the paper:

1. **We averaged coefficients** - Paper says to **route to task-specific coefficients**
2. **We trained full LoRA then projected** - Paper says to **train only coefficients with frozen basis**

Current best result: 83.3% with prompt engineering, 73.3% with Share (averaged).

See `docs/paper-checklists.md` for the full checklist.

---

## Priority Experiments (in order)

### Experiment 1: Routing vs Averaging

**Goal**: Demonstrate that task-specific routing beats coefficient averaging.

**Steps**:
1. Load existing Share basis and coefficients from `runs/share_proper_trained/`
2. Implement error classifier (regex patterns exist in `scripts/routed_inference.py`)
3. At inference time, select coefficient based on error type
4. Run full eval with routing
5. Compare to 73.3% averaged baseline

**Expected**: Routing should outperform averaging significantly.

**Files**:
- `scripts/routed_inference.py` - Has error pattern regexes
- `scripts/share_complete.py` - Has Share algorithm implementation
- `runs/share_proper_trained/` - Saved basis + coefficients

### Experiment 2: Coefficient-Only Training (True Phase 2)

**Goal**: Verify that training only coefficients prevents forgetting.

**Steps**:
1. Freeze basis (β, α) completely
2. Initialize new coefficient (k × p) for ONE failure pattern (e.g., bc_003)
3. Train only the coefficient (~21K params)
4. Eval on ALL 30 tasks
5. Verify no regressions on other 29 tasks

**Expected**: Zero forgetting because basis is frozen.

**Code needed**:
```python
# Key difference from what we did:
# - Freeze basis: basis_B.requires_grad = False
# - Only optimize coefficients: optimizer = Adam([coef_B, coef_A])
# - Coefficient size: (k, p) where p=1, NOT (k, r)
```

### Experiment 3: Sequential Learning Curve

**Goal**: Demonstrate continual learning (train task 1, then 2, then 3...).

**Steps**:
1. Start with base model (no adapters)
2. Train coefficient for task 1 → eval
3. Train coefficient for task 2 → eval (verify task 1 still passes)
4. Repeat for 5-10 tasks
5. Plot learning curve showing no forgetting

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
