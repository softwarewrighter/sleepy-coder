# Product Requirements Document: Sleepy Coder

## Overview

**Project Name**: sleepy-coder
**Tagline**: A continual-learning coding agent that improves overnight

Sleepy Coder is a practical AI coding agent that learns from its mistakes using parameter-efficient continual learning (PaCT). Like humans processing the day's events during sleep, the agent consolidates learnings from daytime coding sessions into long-term improvements during an overnight "sleep" phase.

## Problem Statement

Current AI coding agents:
1. Repeat the same mistakes across sessions
2. Require expensive full fine-tuning to improve
3. Suffer from catastrophic forgetting when learning new skills
4. Accumulate many LoRA adapters that become unmanageable ("adapter zoo")

## Solution

A minimal coding agent (Pi-like loop) with a continual learning subsystem that:
1. Captures failure-to-fix episodes during daytime coding
2. Trains small updates overnight using parameter-efficient methods
3. Consolidates knowledge into a shared low-rank subspace (Share/UWSH-style)
4. Maintains regression safety through strict gating

## Target Users

1. **Developers** wanting a local coding assistant that improves over time
2. **Researchers** exploring continual learning for code generation
3. **Educators** demonstrating PaCT concepts with visual, reproducible demos

## Core Requirements

### 1. Minimal Agent Runtime (Rust)

- Pi-like minimal agent loop with tools: edit, run, patch, search
- Support for local LLM inference (Ollama, vLLM, HF TGI)
- Deterministic episode capture (prompts, errors, diffs, outcomes)
- Sandboxed code execution (cargo check/test in temp directories)

### 2. Rust Koans Task System

- 50-200 tiny Rust tasks (10-40 lines each)
- One deliberate bug per task
- Bug families: borrow checker, trait bounds, lifetimes, Result/?, iterators
- Deterministic "correct" solutions for evaluation

### 3. Episode Capture & Storage

- SQLite database for episodes
- Schema: task_id, attempt_idx, error_signature, diff, passed, steps_to_green
- Export to JSONL for training
- Normalized error signatures for clustering

### 4. Sleep Training Pipeline (Python)

- Curate failed-then-fixed episodes into training pairs
- Cluster mistakes by error family
- Train LoRA adapters on small batches
- Support for Share-style consolidation (Phase 2)
- UWSH-style coefficient-only updates (Phase 2)

### 5. Evaluation & Gating

- Frozen regression suite (must not regress)
- Metrics: repeat-error rate, steps-to-green, pass rate
- Promote adapter only if gates pass
- Rollback mechanism for failed promotions

### 6. Visualization

- Plots: repeat-error rate vs iteration, median steps-to-green, pass rate
- PNG output + optional Streamlit dashboard
- Visual proof of "learning over time"

## Non-Functional Requirements

### Performance
- Run on small GPUs (4-8GB VRAM) with 1B-3B parameter models
- Support for 4-bit quantized inference
- Sleep cycle < 15 minutes on consumer hardware

### Reproducibility
- Fixed seeds for all random operations
- Versioned datasets and skill snapshots
- Reproducible runs with config.yaml per run

### Safety
- Only train on examples that pass tests + clippy
- Weight human-approved patches higher
- Never promote adapters that cause regression

## Success Metrics

1. **Repeat-error rate decreases** over 6-12 sleep cycles
2. **Steps-to-green improves** (fewer tool calls to fix tasks)
3. **Regression suite maintains** >= 95% pass rate
4. **Demo video** showing visual improvement over time

## Deliverables

### Phase 1: MVP
1. Rust koans runner + sandbox
2. Episode logger (SQLite + JSONL)
3. Python curator + LoRA trainer
4. Eval metrics + matplotlib plots
5. scripts/quick_cycle.sh for fast iteration

### Phase 2: PaCT Methods
1. Orthogonal/residual constraint variant
2. Share-style consolidation (basis + coefficients)
3. UWSH-style basis estimation
4. Side-by-side comparison demos

### Phase 3: Production Agent
1. Run on real Rust repos
2. Per-repo coefficient personalization
3. Strong regression gating
4. Integration with existing workflows

## Content Deliverables

1. **YouTube Short** (2 min): "This AI learns from its coding mistakes while it sleeps"
2. **YouTube Explainer** (15-25 min): "How to Build a Coding AI That Learns from Its Mistakes"
3. **Blog Post**: "Sleepy Coder: A Practical Continual-Learning Coding Agent"

## References

- arXiv:2602.06043v1 - "Shared LoRA Subspaces for almost Strict Continual Learning"
- arXiv:2512.05117 - Universal Weight Subspace Hypothesis (UWSH)
- PaCT - Parameter-Efficient Continual Finetuning process
- O-LoRA, KeepLoRA, SPARC, C-LoRA related work
