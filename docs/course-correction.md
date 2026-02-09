# Course Correction: Proper Share Implementation

**Date**: 2026-02-09
**Status**: Planning

## Problem Statement

After 12 training cycles, we have not achieved improvement over baseline. The fundamental issue is that **we have not actually implemented the Share paper properly**.

## Current Results

| Cycle | Training Approach | Pass Rate | vs Baseline |
|-------|------------------|-----------|-------------|
| C0 | None (baseline) | **76.7%** | -- |
| C1 | Naive LoRA | 60.0% | -16.7% |
| C2-3 | With replay | 66-70% | -7 to -10% |
| C9-10 | Minimal (20 steps) | 73.3% | -3.4% |
| C11 | Expanded data (112 ex) | 70.0% | -6.7% |
| C12 | Rust 2024 + replay | 73.3% | -3.4% |

**Best achieved**: 73.3% (3.4% below baseline)

## What We Did vs. What Share Recommends

| Aspect | Share Paper | Our Implementation |
|--------|-------------|-------------------|
| **# of adapters** | 6-50+ distinct tasks | 6 adapters on same data |
| **Algorithm** | SVD → basis extraction → freeze → train coefficients | Just merged adapters |
| **Subspace update** | Incremental SVD merge | None |
| **Replay** | **Not needed** (whole point) | Heavy replay (50%+) |
| **Novel data** | Each task = genuinely distinct domain | Same Rust koans reshuffled |
| **Basis freezing** | Yes - only train coefficients | No - full LoRA every time |

## Share Paper Key Details

From [arXiv:2602.06043](https://arxiv.org/abs/2602.06043):

### Algorithm (3 phases)

1. **Initialization**: Extract k principal basis vectors from N≥1 LoRA adapters using SVD
2. **Continual Adaptation**: Add φ temporary factors when new data arrives; optimize coefficients only
3. **Merging**: Reconstruct prior task adapters, stack with new ones, perform SVD to update principal factors

### Hyperparameters

- **k (principal factors)**: Determined by "60% explained variance" threshold
- **p (pseudo-rank)**: p=1 is effective, higher values yield minimal additional benefits; suggest p=r/3
- **φ (temporary factors)**: Effective range is φ=[1, k/4]

### Scale Tested

- GLUE: 6 tasks
- Image Classification: 40 tasks (4 datasets × 10 tasks)
- 3D Pose: 12 object categories
- Text-to-Image: 44 continual tasks
- **LoRA at Scale: 50 adapters arriving incrementally (from pool of 500)**

### Key Claim

> "A single Share model can replace hundreds of task-specific LoRA adapters"

## Root Causes of Our Failure

1. **Not enough adapters**: We trained 6 adapters on essentially the same data. Share requires distinct tasks to find the shared subspace.

2. **No SVD-based subspace extraction**: We just merged LoRA weights. Share extracts principal directions via SVD and freezes them.

3. **No coefficient-only training**: We retrain full LoRA every cycle. Share freezes the basis and only trains small coefficient vectors.

4. **Not training on genuinely novel data**: Our training data overlaps heavily with what the model already knows. Each adapter should learn something NEW.

5. **Heavy reliance on replay**: Share explicitly avoids replay by using the shared subspace. We used 50%+ replay as a band-aid.

## Corrective Action Plan

### Phase 2.1: Generate Truly Distinct Task Families (10-50 adapters)

Create distinct training sets for each adapter:

1. **Borrow Checker Variants** (10 adapters)
   - bc_move: Move semantics only
   - bc_ref: Reference/borrow patterns
   - bc_mut: Mutable borrow conflicts
   - bc_lifetime_simple: Basic lifetime annotations
   - bc_lifetime_complex: Multi-lifetime scenarios
   - bc_copy_clone: Copy vs Clone decisions
   - bc_rc_arc: Smart pointer patterns
   - bc_cell_refcell: Interior mutability
   - bc_closure_capture: Closure capture modes
   - bc_async_borrow: Async borrow patterns

2. **Trait System** (10 adapters)
   - tb_derive: Derive macro usage
   - tb_impl: Manual trait implementations
   - tb_bounds: Generic bounds
   - tb_where: Where clauses
   - tb_associated: Associated types
   - tb_dyn: Dynamic dispatch
   - tb_send_sync: Thread safety traits
   - tb_from_into: Conversion traits
   - tb_iterator: Iterator trait implementations
   - tb_display_debug: Formatting traits

3. **Error Handling** (10 adapters)
   - rh_option: Option handling patterns
   - rh_result: Result handling patterns
   - rh_question: ? operator usage
   - rh_match: Pattern matching on Result/Option
   - rh_combinators: map, and_then, ok_or, etc.
   - rh_custom_error: Custom error types
   - rh_thiserror: thiserror derive patterns
   - rh_anyhow: anyhow error handling
   - rh_unwrap_expect: Appropriate panic patterns
   - rh_early_return: Early return patterns

4. **Rust 2024 Edition** (10 adapters)
   - r24_fmt: Inline format strings
   - r24_let_chain: Let-chains
   - r24_let_else: Let-else patterns
   - r24_is_some_and: Modern Option methods
   - r24_is_ok_and: Modern Result methods
   - r24_copied: iter().copied() vs cloned()
   - r24_flatten: Iterator::flatten patterns
   - r24_matches: matches! macro
   - r24_clippy_modern: Modern clippy lints
   - r24_async_closure: Async closures

5. **Advanced Patterns** (10 adapters)
   - adv_builder: Builder pattern
   - adv_newtype: Newtype pattern
   - adv_typestate: Typestate pattern
   - adv_phantom: PhantomData usage
   - adv_unsafe: Safe unsafe abstractions
   - adv_macro: Declarative macros
   - adv_proc_macro: Procedural macros
   - adv_simd: SIMD patterns
   - adv_ffi: FFI patterns
   - adv_pin: Pin<T> patterns

### Phase 2.2: Implement Actual Share Algorithm

```python
# scripts/share_consolidate.py

def share_consolidate(adapters: List[Path], k_target_variance: float = 0.6) -> SharedBasis:
    """
    Implement Share Algorithm 1 from arXiv:2602.06043
    """
    # 1. Load all LoRA adapters
    loras = [load_lora_adapter(p) for p in adapters]

    # 2. Extract delta_W matrices (B @ A for each layer)
    deltas = {}
    for layer_name in loras[0].keys():
        deltas[layer_name] = np.stack([
            (lora[layer_name]['B'] @ lora[layer_name]['A']).flatten()
            for lora in loras
        ])

    # 3. Center the matrices
    for layer_name in deltas:
        mean = deltas[layer_name].mean(axis=0)
        deltas[layer_name] = deltas[layer_name] - mean

    # 4. SVD to find shared subspace
    basis = {}
    for layer_name in deltas:
        U, S, Vh = np.linalg.svd(deltas[layer_name], full_matrices=False)

        # Select k based on explained variance
        total_var = (S ** 2).sum()
        cumvar = np.cumsum(S ** 2) / total_var
        k = np.searchsorted(cumvar, k_target_variance) + 1

        basis[layer_name] = {
            'Vh': Vh[:k],  # Top-k principal directions (FROZEN)
            'S': S[:k],    # Singular values
            'k': k
        }

    # 5. Compute per-adapter coefficients
    coefficients = []
    for i, lora in enumerate(loras):
        coef = {}
        for layer_name in basis:
            delta_flat = (lora[layer_name]['B'] @ lora[layer_name]['A']).flatten()
            coef[layer_name] = basis[layer_name]['Vh'] @ delta_flat
        coefficients.append(coef)

    return SharedBasis(basis=basis, coefficients=coefficients)


def train_new_task_coefficients(
    shared_basis: SharedBasis,
    new_data: Dataset,
    base_model: str,
    learning_rate: float = 1e-4,
    steps: int = 100
) -> Coefficients:
    """
    Train ONLY coefficients for a new task (basis is frozen)
    """
    # Initialize coefficients to zero
    coefficients = {
        layer: torch.zeros(shared_basis.basis[layer]['k'])
        for layer in shared_basis.basis
    }

    # Training loop - only optimize coefficients
    # This is MUCH cheaper than full LoRA training
    ...

    return coefficients
```

### Phase 2.3: Incremental Subspace Update

```python
def merge_new_adapter(
    shared_basis: SharedBasis,
    new_adapter: LoraAdapter,
    new_coefficients: Coefficients
) -> SharedBasis:
    """
    Merge a new adapter into the shared subspace (Algorithm 1, Phase 3)
    """
    # 1. Reconstruct all prior adapters from basis + coefficients
    reconstructed = []
    for coef in shared_basis.coefficients:
        adapter = reconstruct_adapter(shared_basis.basis, coef)
        reconstructed.append(adapter)

    # 2. Add new adapter
    reconstructed.append(new_adapter)

    # 3. Re-run SVD to update subspace
    return share_consolidate(reconstructed)
```

### Phase 2.4: Evaluation

After implementing Share properly:

1. Train 10-50 adapters on distinct task families
2. Consolidate using SVD
3. Train new tasks with coefficient-only updates
4. Evaluate on frozen set

**Expected outcome**: Should match or exceed baseline (76.7%) since we're adding knowledge without overwriting.

## Files to Create

1. `scripts/generate_task_families.py` - Generate 50 distinct task sets
2. `scripts/share_consolidate.py` - Implement Share Algorithm 1
3. `scripts/train_coefficients.py` - Coefficient-only training
4. `scripts/share_eval.py` - Evaluation with Share model
5. `cuda/scripts/share_train.py` - CUDA training for Share

## Success Criteria

1. **Pass rate >= 76.7%** (match baseline)
2. **Learning demonstrated** without replay
3. **Single model** replacing 50+ adapters
4. **Coefficient-only updates** for new tasks (cheaper than full LoRA)

## References

- [Share Paper (arXiv:2602.06043)](https://arxiv.org/abs/2602.06043)
- [UWSH (arXiv:2512.05117)](https://arxiv.org/abs/2512.05117)
- [Previous changes](./changes.md)
