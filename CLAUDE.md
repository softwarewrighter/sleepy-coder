# CLAUDE.md - AI Agent Context

This file provides context for AI agents (like Claude) working on this codebase.

## Project Overview

sleepy-coder is a continual learning agent for fixing Rust compilation errors. It uses a day/night cycle where:
- During "day", the agent attempts to fix buggy Rust code
- During "night", failed episodes train the model with LoRA
- Progress is tracked on a frozen evaluation set

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI (sleepy-coder)                     │
├─────────────────────────────────────────────────────────────┤
│  Agent Loop    │    Eval Harness    │   Tasks/Koans       │
├─────────────────────────────────────────────────────────────┤
│  OllamaClient  │    EpisodeStore    │     Sandbox         │
├─────────────────────────────────────────────────────────────┤
│                     Core Types                              │
└─────────────────────────────────────────────────────────────┘
```

## Crates

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| core_types | Shared types | Episode, Task, EvalResult, ErrorFamily |
| capture | Episode storage | EpisodeStore (SQLite) |
| sandbox | Code execution | Sandbox, CompileResult, TestResult |
| tasks_rust_koans | Task definitions | 42 builtin koans, load/filter functions |
| agent | LLM integration | OllamaClient, AgentLoop, AgentResult |
| eval | Evaluation | EvalHarness, EvalMetrics, EvalRun |
| cli | User interface | run, eval, list, show commands |

## Key Patterns

### TDD Development
All code follows Test-Driven Development:
1. RED: Write failing tests first
2. GREEN: Implement to pass tests
3. REFACTOR: Clean up with clippy/fmt

### Error Families
Koans are categorized by error type:
- `BorrowChecker` - Move/borrow errors (E0382, E0502, etc.)
- `Lifetimes` - Lifetime annotations (E0106, E0621)
- `TraitBounds` - Missing traits (E0277)
- `ResultHandling` - Result/Option misuse (E0308)
- `TypeMismatch` - Type errors (E0308, E0369)

### Episode Structure
Each learning episode captures:
- task_id, attempt_idx, model_id
- error_signature (normalized)
- diff_unified, passed, steps_to_green
- Timing and token counts

## Development Commands

```bash
# Build and test
./scripts/build-all.sh
./scripts/test-all.sh
./scripts/lint.sh

# Run CLI
cd rust && cargo run -- list
cd rust && cargo run -- run --count 5 --family borrow
cd rust && cargo run -- eval --cycle 0
```

## Code Style

- Rust 2024 edition with let-chains
- All clippy warnings treated as errors
- No disabling clippy checks
- rustfmt for formatting
- Markdown validation for docs

## CRITICAL: Package Management

**NEVER use `pip` directly. ALWAYS use `uv pip` instead.**

This is required because:
1. `uv` is significantly faster than pip
2. Consistent dependency resolution across environments
3. Project standard - no exceptions

```bash
# WRONG - Never do this
pip install torch
pip show package

# CORRECT - Always do this
uv pip install torch
uv pip show package

# Setup uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Test Coverage

Total: 72 tests across 7 crates
- Integration tests marked `#[ignore]` require Ollama running

## Important Files

- `rust/Cargo.toml` - Workspace configuration
- `docs/status.md` - Current project status
- `docs/plan.md` - Implementation plan
- `docs/architecture.md` - System architecture
- `docs/design.md` - Technical design decisions

## Phase Status

- Phase 0: Setup - Complete
- Phase 1: MVP - Complete (72 tests)
- Phase 2: PaCT - Not started (LoRA training)
- Phase 3: Production - Not started
