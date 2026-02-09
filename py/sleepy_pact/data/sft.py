"""SFT (Supervised Fine-Tuning) dataset preparation."""

import json
from dataclasses import dataclass
from pathlib import Path

from sleepy_pact.data.episodes import Episode


@dataclass
class SFTExample:
    """A single example for supervised fine-tuning."""

    instruction: str
    input_text: str
    output: str
    task_id: str
    steps: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
            "task_id": self.task_id,
            "steps": self.steps,
        }

    def to_chat_format(self) -> list[dict]:
        """Convert to chat format for training."""
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.input_text},
            {"role": "assistant", "content": self.output},
        ]

    def to_alpaca_format(self) -> dict:
        """Convert to Alpaca format."""
        return {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
        }


# Default instruction for code fixing
DEFAULT_INSTRUCTION = """You are a Rust expert. Fix the following code that has a compilation error.
Return ONLY the fixed Rust code without any explanation or markdown formatting."""


def build_sft_example(
    buggy_code: str,
    fixed_code: str,
    error_message: str,
    task_id: str,
    steps: int,
    instruction: str = DEFAULT_INSTRUCTION,
) -> SFTExample:
    """Build an SFT example from buggy/fixed code pair.

    Args:
        buggy_code: The original buggy Rust code.
        fixed_code: The corrected Rust code.
        error_message: The compiler error message.
        task_id: Task identifier.
        steps: Number of steps taken to fix.
        instruction: System instruction.

    Returns:
        SFTExample ready for training.
    """
    input_text = f"""## Buggy Code:
```rust
{buggy_code}
```

## Compiler Error:
{error_message}

## Fixed Code:"""

    return SFTExample(
        instruction=instruction,
        input_text=input_text,
        output=fixed_code,
        task_id=task_id,
        steps=steps,
    )


def build_sft_dataset(
    episodes: list[Episode],
    tasks: dict[str, dict],
    min_steps: int = 1,
    max_steps: int = 5,
) -> list[SFTExample]:
    """Build SFT dataset from successful episodes.

    Args:
        episodes: List of episodes (should be filtered to successful ones).
        tasks: Dictionary mapping task_id to task data with buggy_code, correct_code.
        min_steps: Minimum steps to include (filters out "too easy" fixes).
        max_steps: Maximum steps to include (filters out "failed" attempts).

    Returns:
        List of SFTExample ready for training.
    """
    examples = []

    for ep in episodes:
        # Only use successful episodes within step range
        if not ep.passed:
            continue
        if ep.steps_to_green < min_steps or ep.steps_to_green > max_steps:
            continue

        # Get task data
        task = tasks.get(ep.task_id)
        if not task:
            continue

        buggy_code = task.get("buggy_code", "")
        fixed_code = task.get("correct_code", "")
        error_sig = ep.error_signature or "Unknown error"

        if not buggy_code or not fixed_code:
            continue

        example = build_sft_example(
            buggy_code=buggy_code,
            fixed_code=fixed_code,
            error_message=error_sig,
            task_id=ep.task_id,
            steps=ep.steps_to_green,
        )
        examples.append(example)

    return examples


def save_sft_dataset(examples: list[SFTExample], path: Path, format: str = "jsonl"):
    """Save SFT dataset to file.

    Args:
        examples: List of SFTExample.
        path: Output file path.
        format: Output format ("jsonl", "alpaca", "chat").
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ex in examples:
            if format == "alpaca":
                data = ex.to_alpaca_format()
            elif format == "chat":
                data = {"messages": ex.to_chat_format()}
            else:  # jsonl
                data = ex.to_dict()

            f.write(json.dumps(data) + "\n")


def load_sft_dataset(path: Path) -> list[SFTExample]:
    """Load SFT dataset from JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of SFTExample.
    """
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                examples.append(
                    SFTExample(
                        instruction=data.get("instruction", DEFAULT_INSTRUCTION),
                        input_text=data.get("input", ""),
                        output=data.get("output", ""),
                        task_id=data.get("task_id", ""),
                        steps=data.get("steps", 0),
                    )
                )
    return examples
