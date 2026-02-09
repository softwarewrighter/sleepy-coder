"""Training configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration for LoRA training."""

    # Model settings
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    output_dir: Path = field(default_factory=lambda: Path("runs/adapters"))

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 means use epochs
    max_seq_length: int = 2048

    # Optimizer settings
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_4bit: bool = True
    use_8bit: bool = False

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Checkpoint format - use safetensors for security (no pickle exploits)
    save_safetensors: bool = True

    # Seed
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "base_model": self.base_model,
            "output_dir": str(self.output_dir),
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "max_seq_length": self.max_seq_length,
            "weight_decay": self.weight_decay,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_4bit": self.use_4bit,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Create from dictionary."""
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])
        return cls(**data)

    @classmethod
    def for_quick_test(cls) -> "TrainConfig":
        """Configuration for quick testing."""
        return cls(
            num_epochs=1,
            max_steps=50,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=5,
            save_steps=25,
        )

    @classmethod
    def for_overnight(cls) -> "TrainConfig":
        """Configuration for overnight training."""
        return cls(
            num_epochs=5,
            batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=200,
        )
