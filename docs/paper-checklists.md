# Paper Implementation Checklists

This document provides step-by-step checklists for demonstrating the concepts from the Share and UWSH papers.

---

## Share Paper (arXiv:2602.06043): Shared LoRA Subspaces

### Phase 1: Initialization (Extracting Shared Basis)

- [x] Stack B matrices from multiple adapters: `B = [B₁, B₂, …, B_t]`
- [x] Stack A matrices from multiple adapters: `A = [A₁, A₂, …, A_t]`
- [x] Mean-center the stacked matrices
- [x] Perform SVD decomposition
- [x] Select k components using 60% explained variance threshold
- [x] Store shared basis (β, α) as orthonormal matrices
- [x] Compute initial coefficients via projection: `ε = β^T B_centered`

**Status**: Complete - we have 51+ adapters consolidated into shared basis

### Phase 2: Coefficient-Only Training (KEY for preventing forgetting)

- [x] Freeze basis vectors (β, α) completely
- [x] Initialize new coefficients for each task: `ε ~ N(0, σ²)`
- [x] Train ONLY coefficients (k × p parameters, typically p=1)
- [x] Verify parameter count: should be ~460x fewer than full LoRA
- [ ] **Verify coefficient dimensions are (k, p) NOT (k, r)**
- [ ] **Measure forgetting rate with coefficient-only training**
- [ ] **Compare coefficient-only vs full adapter training side-by-side**

**Status**: Partially done - need verification experiments

### Phase 3: Merging and Basis Updates

- [x] Reconstruct adapters from basis + coefficients
- [x] Stack with any new adapters
- [x] Re-compute SVD for updated basis
- [x] Re-project all coefficients to new basis
- [ ] **Optionally finetune coefficients after merge**

**Status**: Complete

### Inference Strategy (CRITICAL - our main gap)

- [ ] **Implement task/error classifier**
  - Detect error type from compiler output
  - Map to coefficient set
- [ ] **Route to appropriate coefficients at inference time**
  - Do NOT average coefficients
  - Select single best-matching coefficient set
- [ ] **Compare routed vs averaged inference**
  - Hypothesis: routing > averaging significantly
- [ ] **Evaluate per-error-family pass rates**

**Status**: NOT DONE - we averaged instead of routing

### Experiments to Run

| # | Experiment | Expected Outcome | Status |
|---|------------|------------------|--------|
| 1 | Coefficient-only training (7 failure patterns) | No regressions on other 23 tasks | Not done |
| 2 | Task routing at inference | Higher pass rate than averaging | Not done |
| 3 | Comparison: phase2 vs full LoRA | Phase2 has less forgetting | Partially done |
| 4 | Vary k (10, 20, 30, 40) | Find optimal basis size | Not done |
| 5 | Vary p (1, 2, 4) | Compare coefficient ranks | Not done |
| 6 | Sequential task learning | Each new task doesn't hurt previous | Not done |

---

## UWSH Paper (arXiv:2512.05117): Universal Weight Subspace Hypothesis

### Core Hypothesis Verification

- [x] Extract subspace from diverse adapters
- [ ] **Verify subspace stability across training runs**
  - Train same task from different initializations
  - Compare resulting subspaces via Grassmann distance
- [ ] **Measure explained variance with varying k**
  - Plot cumulative variance vs k
  - Identify "elbow" point
- [ ] **Compare subspaces across error families**
  - Are borrow checker and trait bound subspaces similar?

### Subspace Analysis Experiments

| # | Experiment | Expected Outcome | Status |
|---|------------|------------------|--------|
| 1 | SVD of borrow checker adapters only | Concentrated variance (few k needed) | Not done |
| 2 | SVD of trait bound adapters only | Similar concentration | Not done |
| 3 | Cross-family subspace similarity | High overlap (UWSH predicts ~80%+) | Not done |
| 4 | Random initialization comparison | Similar subspaces emerge | Not done |
| 5 | Different base models | Compare Qwen vs CodeLlama subspaces | Not done |

### Practical Applications

- [ ] **Minimal k experiment**: Find smallest k that maintains 95% of full-adapter performance
- [ ] **Transfer learning**: Train coefficient on one error, test on similar error
- [ ] **Zero-shot coefficient**: Use nearest-neighbor coefficient for unseen error types

---

## Combined Workflow (Share + UWSH)

### Step 1: Establish Strong Basis (Week 1)

1. [ ] Collect diverse high-quality LoRA adapters (aim for 20-50)
2. [ ] Run Phase 1 to extract shared basis
3. [ ] Analyze explained variance curve
4. [ ] Select k using UWSH-informed threshold (60% or elbow)
5. [ ] Verify basis orthonormality

### Step 2: Coefficient Training Pipeline (Week 2)

1. [ ] Implement proper Phase 2 training loop
   - Input: Task data + frozen basis
   - Output: Coefficient matrix (k × p)
   - Loss: Standard LM loss
2. [ ] Train coefficients for each failure pattern
3. [ ] Verify no forgetting on held-out tasks

### Step 3: Routing System (Week 3)

1. [ ] Build error classifier (regex or embeddings)
2. [ ] Map error types to coefficient indices
3. [ ] Implement routing at inference time
4. [ ] Fallback: Use "general" coefficient when uncertain

### Step 4: Evaluation and Comparison (Week 4)

1. [ ] Full eval with routing
2. [ ] Compare: routing vs averaging vs baseline
3. [ ] Per-family breakdown
4. [ ] Forgetting analysis (continual learning curve)

---

## What We Did vs What Papers Recommend

| Paper Says | What We Did | Gap |
|------------|-------------|-----|
| Phase 2 trains only coefficients | Trained full LoRA, then projected | Need true Phase 2 |
| Coefficient size k × p (p ≈ 1) | Stored k × r coefficients | Wrong dimensions |
| Task-specific routing at inference | Averaged coefficients | Need router |
| 60% variance threshold for k | Used this | Done |
| Orthonormal basis from SVD | Used this | Done |
| Sequential task training | Trained on all patterns at once | Need sequential |

---

## Priority Order for Demonstrations

### High Priority (Core Claims)

1. **Routed inference beats averaging**
   - This is the most direct demonstration of Share's value
   - Should show immediate improvement over our 73.3% average result

2. **Coefficient-only training prevents forgetting**
   - Train on ONE new pattern
   - Verify other 29 patterns still pass
   - Compare to full LoRA (which caused regressions)

### Medium Priority (Supporting Evidence)

3. **Parameter efficiency**
   - Document: 1.6M params (full LoRA) vs ~21K (Share coefficients)
   - Show similar performance with 76x fewer parameters

4. **Subspace stability (UWSH)**
   - Show that different adapter sets converge to similar subspaces
   - Validates the theoretical foundation

### Lower Priority (Nice to Have)

5. **Cross-family transfer**
   - Can a coefficient trained on E0502 help with E0382?

6. **Minimal k search**
   - How low can k go before performance drops?

---

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|--------------|
| Pass rate with routing | ≥ 80% | 73.3% (averaged) |
| Forgetting rate | 0% | ~10% with full LoRA |
| Parameter reduction | ≥ 50x | 76x (coefficient only) |
| Subspace overlap | ≥ 70% | Not measured |

---

## Technical Requirements

### For Coefficient-Only Training

```python
# Pseudo-code for proper Phase 2
def train_phase2(basis_B, basis_A, task_data, k, p=1):
    # Initialize small coefficient matrices
    coef_B = torch.randn(k, p) * 0.01
    coef_A = torch.randn(k, p) * 0.01

    # Freeze basis
    basis_B.requires_grad = False
    basis_A.requires_grad = False

    # Only coefficients are trainable
    optimizer = Adam([coef_B, coef_A], lr=1e-4)

    for batch in task_data:
        # Reconstruct adapter from basis + coefficients
        B_adapter = basis_B @ coef_B  # (n, p)
        A_adapter = basis_A @ coef_A  # (d, p)

        # Forward pass with reconstructed adapter
        loss = model.forward_with_adapter(batch, B_adapter, A_adapter)
        loss.backward()
        optimizer.step()

    return coef_B, coef_A
```

### For Routing

```python
# Pseudo-code for task routing
def select_coefficient(error_message, coefficients_map):
    error_type = classify_error(error_message)

    if error_type in coefficients_map:
        return coefficients_map[error_type]
    else:
        return coefficients_map["general"]  # fallback
```

---

## Next Actions

1. [ ] Implement proper routing system for existing coefficients
2. [ ] Run eval with routing instead of averaging
3. [ ] Document results in comparison table
4. [ ] If routing helps, implement proper Phase 2 training
5. [ ] Create sequential learning experiment (train task 1, eval, train task 2, eval, ...)
