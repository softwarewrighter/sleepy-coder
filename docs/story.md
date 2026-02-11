# The Sleepy Coder Story

## Can We Teach a Small AI to Fix Rust Compiler Errors?

---

## The Dream

What if an AI could learn from its mistakes while you sleep?

We set out to build **Sleepy Coder** - a 1.5 billion parameter model that fixes Rust compilation errors. The twist? It learns continuously, improving overnight from every failure it encounters.

The goal was simple: beat the baseline, prove continual learning works, and do it with a model small enough to run on a single GPU.

---

## Chapter 1: The First Setback

### "More Data Should Work... Right?"

We started the obvious way - collect training data, fine-tune the model, measure results.

**Baseline: 73.3%** (22 out of 30 tasks)

We trained. We added more data. We trained again.

**Result: 73.3%**

No improvement. The model memorized examples but couldn't generalize.

> **Lesson learned:** Throwing more data at a problem doesn't always help.

---

## Chapter 2: The Paper Chase

### "There Has to Be a Better Way"

We discovered two research papers that changed everything:

1. **UWSH (December 2025)** - Neural networks converge to shared low-dimensional subspaces
2. **Share (February 2026)** - A practical algorithm for continual learning

The Share paper claimed we could add new skills to a model with **76x fewer parameters** per task.

Sounds too good to be true? We thought so too.

---

## Chapter 3: The Big Misunderstanding

### "We Implemented It Wrong"

Our first Share implementation:
1. Train full LoRA adapters (~1.6M parameters each)
2. Combine them with Share algorithm
3. Profit?

**Result: 70%** - Actually WORSE than baseline.

We went back to the paper. Read it again. And again.

Then we found it - one line that changed everything:

> *"Train ONLY the coefficients. Keep the basis FROZEN."*

We had been training full adapters. The paper said train tiny coefficient vectors (~21K parameters).

**76x smaller. We missed the whole point.**

---

## Chapter 4: The Real Implementation

### "Finally Getting It Right"

We rebuilt everything from scratch:

**Phase 1:** Extract a shared basis from existing adapters using SVD
- Mathematical magic: find the common "directions" all adapters share

**Phase 2:** For new tasks, train ONLY tiny coefficient vectors
- 21,000 parameters instead of 1,600,000
- Same results, fraction of the cost

**Phase 3:** Periodically update the shared basis as knowledge grows

**Result: 76.7%** - Our best score ever!

> **+3.4% improvement over baseline. One more task solved.**

---

## Chapter 5: The Scaling Trap

### "If 59 Adapters Work, 100 Must Be Better!"

Drunk on success, we pushed further.

- 59 adapters → 76.7%
- Let's try 81 adapters!

**Result: 70%**

Wait, what? MORE adapters made it WORSE?

The math was cruel but clear: when you compress too many adapters into a shared space, specialized knowledge gets diluted. It's like mixing too many paint colors - you just get brown.

> **Lesson learned:** Quality beats quantity. Sometimes less is more.

---

## Chapter 6: The Stubborn Seven

### "Some Problems Just Won't Budge"

No matter what we tried, 7 tasks kept failing:

| Task | The Problem |
|------|------------|
| bc_003 | Mutable borrow while immutable exists |
| bc_005 | Double mutable borrow |
| bc_010 | Returning reference to local data |
| tb_002 | Missing Clone trait |
| tb_007 | Missing Hash trait |
| tb_008 | Missing Ord trait |
| rh_004 | Option to Result conversion |

These are **hard** Rust patterns. They require understanding:
- Ownership semantics
- Lifetime rules
- When to derive vs implement traits

We tried:
- More training data ❌
- Targeted examples ❌
- Distillation from larger models ❌
- Different learning rates ❌

**Nothing worked.**

---

## Chapter 7: The Averaging Trap

### "Blend the Best, Get the Worst"

We had coefficients for dozens of tasks. Logical next step: average them to get a "universal" model.

Every. Single. Time:

- Task-specific coefficients: 76.7%
- Averaged coefficients: 73.3%

Averaging destroys specialization. The paper even warned us:

> *"Task-specific coefficients outperform averaged ones."*

We learned this the hard way. Multiple times.

---

## Chapter 8: Running Out of Disk Space

### "The Experiments Are Eating My SSD"

Plot twist nobody asked for: we filled up 100GB with experiment artifacts.

Old models, checkpoints, failed runs - all piling up while we focused on the science.

**3 AM emergency cleanup session.** Deleted 86GB of old experiments.

> **Lesson learned:** Science needs good housekeeping too.

---

## Where We Are Now

### The Scoreboard

| What | Result |
|------|--------|
| **Best achieved** | 76.7% (23/30 tasks) |
| **Baseline** | 73.3% (22/30 tasks) |
| **Improvement** | +1 task, +3.4% |
| **Parameter reduction** | 76x per new task |

### What Actually Works

✅ Share algorithm with proper coefficient-only training
✅ Task-specific coefficients (never average!)
✅ Quality over quantity for adapter diversity
✅ SVD-based subspace extraction

### What We Learned the Hard Way

❌ More data doesn't fix everything
❌ More adapters can hurt performance
❌ Averaging destroys specialization
❌ Small models have real limits

---

## The Limitations

### Being Honest About What We Can't Do

A 1.5B parameter model hitting a wall on complex Rust patterns isn't a bug - it's a feature of model scale.

These 7 tasks require:
- Deep semantic understanding of ownership
- Multi-step reasoning about lifetimes
- Knowledge of Rust idioms and patterns

**A small model can learn patterns. It struggles with principles.**

---

## What's Next

### The Road Ahead

**Near term:**
1. Implement routed inference - select the right coefficients based on the error type
2. Test on larger base models (3B, 7B parameters)
3. Hybrid approach - use Claude/GPT-4 for hard cases, fine-tuned model for common ones

**Longer term:**
1. Curriculum learning - start simple, build complexity
2. Chain-of-thought training - teach reasoning, not just patterns
3. Real-world deployment - VS Code extension?

---

## The Takeaway

### What This Project Taught Us

1. **Read papers carefully** - one missed detail can waste weeks
2. **Simple baselines matter** - always know what you're beating
3. **Failure is data** - each setback taught us something
4. **Model size has limits** - some problems need bigger brains
5. **Science is messy** - the path isn't straight, and that's okay

---

## The Numbers That Matter

```
Parameters saved per task:  76x
Best pass rate achieved:    76.7%
Tasks that still fail:      7
Lessons learned:            Countless
```

---

## Want to Try It?

The code is open source. The model runs on a single GPU.

```bash
# Evaluate the model
cargo run -- eval --model sleepy-coder --cycle 1

# See it try to fix Rust errors
cargo run -- run --count 5
```

---

*Built with frustration, coffee, and occasionally working code.*

*Sleepy Coder - Learning from mistakes so you don't have to.*
