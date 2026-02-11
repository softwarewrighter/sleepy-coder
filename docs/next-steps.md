# Next Steps: Scaling Share and Continual Learning

**Date**: 2026-02-10
**Status**: Active

---

## Current State

| Metric | Value |
|--------|-------|
| Baseline Pass Rate | 73.3% |
| Best Share Result | 73.3% (no regression, no improvement) |
| Available Adapters | 148 (51 unique patterns) |
| Training Examples | ~1,045 total |
| Persistent Failures | 8 koans |

### Persistent Failures (never fixed by any approach)
- bc_003: mut_borrow_conflict
- bc_005: double_mut_borrow
- bc_010: return_local_ref
- rh_004: option_ok_or
- rh_005: result_map_err
- tb_002: missing_clone
- tb_007: missing_hash
- tb_008: missing_ord

---

## Strategy 1: Scale to 100+ Diverse Adapters

### Why
The Share paper (arXiv:2602.06043) uses 50-150 adapters. More adapters = richer shared subspace = more expressive continual learning.

### Current
- 51 pattern-specific adapters (trained on 51 code patterns)
- 6 domain-specific adapters (yew, axum, sqlx, cli, refactoring, style)

### Action Plan
```bash
# 1. Train adapters on MORE distinct patterns
# Generate 50 new patterns from the 8 failure cases
python scripts/generate_failure_variants.py --count 50

# 2. Train an adapter for each
for pattern in data/sft/patterns/failure_variants/*.jsonl; do
    python cuda/scripts/train.py --data "$pattern" --output runs/adapters/variants/$(basename $pattern .jsonl)
done

# 3. Rebuild Share with 100+ adapters
python scripts/share_full_algorithm.py phase1 --adapters runs/adapters/all_100 --output runs/share100
```

---

## Strategy 2: LLM Distillation (Knowledge Transfer)

### Why
A large LLM (Claude, GPT-4) knows how to fix these Rust errors. Use it to generate training data for the small model.

### How It Works
1. Large LLM generates (buggy_code, error, fixed_code) triples
2. Small model learns from these examples
3. Knowledge is "distilled" into the small model

### Implementation
```python
# scripts/distill_from_llm.py
import anthropic

client = anthropic.Client()

FAILURE_PATTERNS = [
    "mut_borrow_conflict",
    "double_mut_borrow",
    "return_local_ref",
    # ... etc
]

def generate_training_pair(pattern: str) -> dict:
    """Use Claude to generate a training example."""
    prompt = f"""Generate a Rust code example that demonstrates the "{pattern}" error.

Output JSON with:
- buggy_code: The code with the error
- error_message: The rustc error
- fixed_code: The corrected code
- explanation: Why this fix works

Make the example realistic and non-trivial."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_response(response)

# Generate 100 examples per failure pattern
for pattern in FAILURE_PATTERNS:
    examples = [generate_training_pair(pattern) for _ in range(100)]
    save_jsonl(examples, f"data/sft/distilled/{pattern}.jsonl")
```

### Expected Outcome
- 800+ high-quality training examples (100 × 8 patterns)
- Directly aligned with failure cases
- Diverse code structures and contexts

---

## Strategy 3: Incremental Continual Learning

### Why
Share is designed for continual learning. Instead of one-shot training, keep adding knowledge in cycles.

### The Day/Night Cycle
```
Day:   Agent attempts to fix koans
       ↓
       Some fail → capture error + wrong attempt
       ↓
Night: Train new coefficients on failures (Phase 2)
       ↓
       Merge into Share model (Phase 3)
       ↓
       Repeat
```

### Implementation
```bash
# Automated continual learning loop
for cycle in $(seq 1 100); do
    echo "=== Cycle $cycle ==="

    # Day: Eval and capture failures
    cargo run -q --release -- eval --model sleepy-share --cycle $cycle

    # Extract failed attempts
    python scripts/extract_failures.py --cycle $cycle --output data/sft/cycle_${cycle}_failures.jsonl

    # Night: Train on failures (Phase 2)
    python scripts/share_full_algorithm.py phase2 \
        --share runs/share_current \
        --data data/sft/cycle_${cycle}_failures.jsonl \
        --task-id cycle_$cycle

    # Merge (Phase 3)
    python scripts/share_full_algorithm.py phase3 \
        --share runs/share_current \
        --trained runs/phase2_cycle_$cycle \
        --task-id cycle_$cycle \
        --output runs/share_current

    # Export and test
    python scripts/share_full_algorithm.py export \
        --share runs/share_current \
        --task averaged \
        --output runs/adapters/cycle_$cycle
done
```

---

## Strategy 4: Learning from Mistakes in Practical Use

### The Corrections Intake System
Already designed in the codebase! See `data/sft/corrections.jsonl`.

### How It Works
1. User runs sleepy-coder on real code
2. Model attempts fix, fails
3. User provides correct fix
4. (wrong_attempt, correct_fix) pair is captured
5. Nightly training incorporates this

### Implementation
```python
# In the agent loop, capture corrections
def capture_correction(task_id: str, wrong_attempt: str, correct_fix: str):
    """Log a user-provided correction for later training."""
    entry = {
        "task_id": task_id,
        "wrong_attempt": wrong_attempt,
        "correct_fix": correct_fix,
        "timestamp": datetime.now().isoformat(),
        "source": "user_correction"
    }
    with open("data/sft/corrections.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
```

### Nightly Training Job
```bash
# Run after each day of practical use
python scripts/train_on_corrections.py
```

---

## Recommended Priority

### Immediate (This Session)
1. **Implement LLM distillation** - Generate 100 examples per failure pattern using Claude
2. **Train adapters** - One per failure pattern with distilled data

### Short Term (Next Session)
3. **Rebuild Share100** - Combine all adapters (51 patterns + 8 failure-specific + others)
4. **Implement task routing** - Use error type to select specific coefficients instead of averaging

### Medium Term
5. **Automate day/night cycle** - Continuous learning from eval failures
6. **Add corrections intake** - Learn from user-provided fixes

---

## Key Insight from Today

> **Averaging coefficients returns to baseline behavior.**
>
> The Share model stores per-task coefficients. When we average them for inference, we lose specialization. The solution is **task routing**: detect error type → select appropriate task coefficients → apply specialized fix.

---

## Files to Create

| Script | Purpose |
|--------|---------|
| `scripts/distill_from_llm.py` | Generate training data using Claude |
| `scripts/generate_failure_variants.py` | Create variations of failure patterns |
| `scripts/extract_failures.py` | Extract failed attempts from eval logs |
| `scripts/train_on_corrections.py` | Train on user-provided corrections |
| `scripts/task_router.py` | Route errors to appropriate Share coefficients |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Pass Rate | 73.3% | ≥80% |
| Adapters in Share | 51 | 100+ |
| Training Examples | 1,045 | 5,000+ |
| Failure Patterns Covered | 8/8 | 8/8 with 100+ examples each |
