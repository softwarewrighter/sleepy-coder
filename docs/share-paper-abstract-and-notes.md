# Share Paper: Abstract and Implementation Notes

**Paper**: "Shared LoRA Subspaces for almost Strict Continual Learning"
**arXiv**: [2602.06043v1](https://arxiv.org/abs/2602.06043)
**Date Fetched**: 2026-02-10

---

## Abstract Summary

Share is a method for continual learning with LoRA that:
- Learns and dynamically updates a **single, shared low-rank subspace** across tasks
- Identifies **essential subspace directions** through SVD
- Achieves **100x parameter reduction** compared to maintaining separate adapters
- Supports **cross-modal** applications

The key innovation is separating the static (shared basis) from dynamic (per-task coefficients) components of LoRA adapters.

---

## Algorithm Details (from paper)

### Notation

- **B matrices**: (n × r) - LoRA's "up" projection
- **A matrices**: (r × d) - LoRA's "down" projection
- **β**: (n × k) - Shared basis for B matrices
- **α**: (d × k) - Shared basis for A matrices
- **ε_β**: (k × p) - Per-task coefficients for B
- **ε_α**: (k × p) - Per-task coefficients for A
- **k**: Number of principal components (chosen by 60% explained variance)
- **p**: Pseudo-rank (typically 1)

### Phase 1: Initialization

Extract foundational subspace from t ≥ 1 available LoRA adapters:

1. Stack B matrices: `ℬ = [B₁, B₂, …, B_t] ∈ ℝ^(n × tr)`
2. Stack A matrices: `𝒜 = [A₁, A₂, …, A_t] ∈ ℝ^(d × tr)`
3. Mean-center and perform SVD
4. Take top-k left singular vectors as basis (60% explained variance threshold)
5. Initialize coefficients with dimensions k × p

### Phase 2: Continual Adaptation (KEY FOR PREVENTING FORGETTING)

When receiving new task data or adapters:

1. Expand temporarily using top φ factors (φ < k)
2. Initialize new coefficients randomly: `ε ~ 𝒩(0, σ²) ∈ ℝ^(φ × p)`
3. **Train ONLY the coefficients** with basis frozen
4. Both temporary basis vectors and coefficients are optimized before merging

**Critical**: This trains orders of magnitude fewer parameters than full LoRA, which is why it prevents catastrophic forgetting.

### Phase 3: Merging and Finetuning

1. Reconstruct all previous task adapters from current basis + coefficients
2. Stack with new adapters
3. Perform SVD to extract updated principal factors
4. Analytically recalculate all coefficients
5. Optionally finetune coefficients for enhanced performance

### Coefficient Computation

Using Moore-Penrose pseudoinverse:
```
ε_β^i = ((β^T β)^(-1) β^T) B̂_i
```

When basis has orthonormal columns (SVD guarantees this), simplifies to:
```
ε_β^i = β^T B̂_i
```

### Inference

The forward pass for task i:
```
h = W₀x + (β ε_β^i)(α ε_α^i)^T x
```

Where you **select the appropriate task's coefficient index i**.

**IMPORTANT**: The paper assumes you know which task you're doing at inference time. For general-purpose models, you need either:
1. A task classifier to select coefficients
2. Train a "general" coefficient set that works across tasks
3. Average coefficients (not explicitly addressed in paper)

---

## Implementation vs Paper Comparison

### Our Implementation Issues Found (2026-02-10)

1. **Coefficient size wrong**: We stored (k, r) but paper says (k, p) with p ≈ 1
2. **Full adapter training**: We trained full LoRA adapters then consolidated, rather than training only coefficients in Phase 2
3. **Inference approach**: We averaged coefficients, but paper says use task-specific selection

### Correct Approach

1. Start with baseline (no LoRA)
2. Train ONE initial adapter on first failing pattern (or few)
3. Extract shared basis (Phase 1)
4. For each additional failing pattern, train ONLY coefficients with frozen basis (Phase 2)
5. Periodically merge to update basis (Phase 3)

### Parameter Comparison

| Component | Full LoRA | Share Phase 2 |
|-----------|-----------|---------------|
| B matrix per layer | n × r | 0 (frozen) |
| A matrix per layer | r × d | 0 (frozen) |
| Coefficients per layer | - | 2 × k × p |
| **Total (L=28, r=8, n=1536, d=1536)** | ~689K per task | ~1.5K per task (with k=26, p=1) |

This is ~460x fewer parameters per new task!

---

## Key Insights

1. **Why Share prevents forgetting**: By freezing the basis and training only tiny coefficient matrices, most parameters remain unchanged. The coefficients act as "addresses" into the shared subspace rather than overwriting knowledge.

2. **Why our approach failed**: Training full LoRA adapters ALREADY causes forgetting before consolidation. Share's benefit requires using Phase 2 (coefficient-only training) from the start.

3. **Storage efficiency**: One shared basis + small coefficient matrices per task is much cheaper than T full adapters.

---

## Next Steps for Our Implementation

1. Use existing 51 adapters to establish a strong shared basis (Phase 1)
2. For new/failing patterns, implement Phase 2 coefficient-only training
3. Implement task routing or general coefficient training for inference
4. Evaluate whether coefficient-only training prevents forgetting
