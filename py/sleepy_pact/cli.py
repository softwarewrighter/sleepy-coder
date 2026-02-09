"""CLI for sleepy-pact training pipeline."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sleepy-pact")


def cmd_prepare(args):
    """Prepare SFT dataset from episodes."""
    from sleepy_pact.data import load_episodes_jsonl, build_sft_dataset, save_sft_dataset

    logger.info(f"Loading episodes from {args.episodes}")

    # Load episodes
    episodes = load_episodes_jsonl(Path(args.episodes))
    logger.info(f"Loaded {len(episodes)} episodes")

    # Load tasks
    tasks = {}
    if args.tasks:
        with open(args.tasks) as f:
            task_list = json.load(f)
            for task in task_list:
                tasks[task["id"]] = task
    logger.info(f"Loaded {len(tasks)} tasks")

    # Filter successful episodes
    successful = [e for e in episodes if e.passed]
    logger.info(f"Found {len(successful)} successful episodes")

    # Build SFT dataset
    examples = build_sft_dataset(successful, tasks)
    logger.info(f"Built {len(examples)} SFT examples")

    # Save
    output_path = Path(args.output)
    save_sft_dataset(examples, output_path, format=args.format)
    logger.info(f"Saved to {output_path}")


def cmd_train(args):
    """Train LoRA adapter."""
    from sleepy_pact.train import TrainConfig, train_lora

    logger.info(f"Training from {args.data}")

    config = TrainConfig(
        base_model=args.model,
        output_dir=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    if args.quick:
        config = TrainConfig.for_quick_test()
        config.base_model = args.model
        config.output_dir = Path(args.output)

    metrics = train_lora(
        sft_data_path=Path(args.data),
        output_dir=Path(args.output),
        config=config,
    )

    logger.info(f"Training complete. Loss: {metrics['train_loss']:.4f}")
    logger.info(f"Adapter saved to {args.output}")


def cmd_plot(args):
    """Generate plots from metrics."""
    from sleepy_pact.viz import generate_all_plots

    logger.info(f"Generating plots from {args.metrics}")

    generate_all_plots(
        metrics_path=Path(args.metrics),
        output_dir=Path(args.output),
    )


def cmd_export_ollama(args):
    """Export trained adapter for Ollama."""
    logger.info(f"Exporting adapter from {args.adapter} for Ollama")

    # This would create a Modelfile and use ollama create
    adapter_path = Path(args.adapter)
    output_name = args.name or f"sleepy-coder-{datetime.now().strftime('%Y%m%d')}"

    modelfile_content = f"""FROM {args.base_model}
ADAPTER {adapter_path}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""

    modelfile_path = adapter_path / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    logger.info(f"Created {modelfile_path}")
    logger.info(f"Run: ollama create {output_name} -f {modelfile_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sleepy-PACT: Training pipeline for sleepy-coder",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare SFT dataset")
    prep_parser.add_argument("--episodes", "-e", required=True, help="Episodes JSONL file")
    prep_parser.add_argument("--tasks", "-t", help="Tasks JSON file")
    prep_parser.add_argument("--output", "-o", default="data/sft/train.jsonl", help="Output file")
    prep_parser.add_argument("--format", "-f", default="jsonl", choices=["jsonl", "alpaca", "chat"])

    # train command
    train_parser = subparsers.add_parser("train", help="Train LoRA adapter")
    train_parser.add_argument("--data", "-d", required=True, help="SFT data JSONL file")
    train_parser.add_argument("--output", "-o", default="runs/adapters", help="Output directory")
    train_parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--quick", action="store_true", help="Quick test mode")

    # plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.add_argument("--metrics", "-m", required=True, help="Metrics JSONL file")
    plot_parser.add_argument("--output", "-o", default="viz", help="Output directory")

    # export command
    export_parser = subparsers.add_parser("export-ollama", help="Export for Ollama")
    export_parser.add_argument("--adapter", "-a", required=True, help="Adapter directory")
    export_parser.add_argument("--base-model", "-m", default="qwen2.5-coder:1.5b")
    export_parser.add_argument("--name", "-n", help="Ollama model name")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "export-ollama":
        cmd_export_ollama(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
