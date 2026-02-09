//! CLI entry point for sleepy-coder.
//!
//! A continual learning agent for fixing Rust code errors.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use core_types::ErrorFamily;

/// sleepy-coder: A continual learning agent for Rust
#[derive(Parser)]
#[command(name = "sleepy-coder")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the agent on koans
    Run {
        /// Number of koans to run (default: all)
        #[arg(short, long)]
        count: Option<usize>,

        /// Filter by error family (borrow_checker, lifetimes, trait_bounds, result_handling, type_mismatch)
        #[arg(short, long)]
        family: Option<String>,

        /// Maximum attempts per task
        #[arg(short = 'a', long, default_value = "5")]
        max_attempts: u32,

        /// Model to use
        #[arg(short, long, default_value = "qwen2.5-coder:1.5b-instruct-q4_K_M")]
        model: String,

        /// Run ID for tracking
        #[arg(long)]
        run_id: Option<String>,
    },

    /// Run evaluation on the frozen eval set
    Eval {
        /// Cycle number for tracking progress
        #[arg(short, long, default_value = "0")]
        cycle: u32,

        /// Maximum attempts per task
        #[arg(short = 'a', long, default_value = "5")]
        max_attempts: u32,

        /// Model to use
        #[arg(short, long, default_value = "qwen2.5-coder:1.5b-instruct-q4_K_M")]
        model: String,

        /// Save episodes to this directory
        #[arg(long, default_value = "data/episodes")]
        save_episodes: String,
    },

    /// List available koans
    List {
        /// Filter by error family
        #[arg(short, long)]
        family: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show a single koan by ID
    Show {
        /// Koan ID
        id: String,
    },

    /// Export episodes and tasks to JSONL for training
    Export {
        /// Output directory
        #[arg(short, long, default_value = "data/export")]
        output: String,

        /// Export only failed episodes (for training data)
        #[arg(long)]
        failed_only: bool,

        /// Export tasks definition
        #[arg(long)]
        tasks: bool,
    },

    /// Run the training pipeline (sleep = learn from mistakes)
    Sleep {
        /// Path to episodes JSONL for training
        #[arg(short, long)]
        episodes: Option<String>,

        /// Path to tasks JSON file
        #[arg(short, long)]
        tasks: Option<String>,

        /// Output directory for trained adapter
        #[arg(short, long, default_value = "runs/adapters")]
        output: String,

        /// Quick test mode (minimal training for testing)
        #[arg(long)]
        quick: bool,

        /// Base model to fine-tune
        #[arg(short, long, default_value = "Qwen/Qwen2.5-Coder-1.5B-Instruct")]
        model: String,
    },

    /// Backup data to text files (for Dropbox sync)
    Backup {
        /// Output directory for backup files
        #[arg(short, long, default_value = "backups")]
        output: String,

        /// Include model checkpoints in backup
        #[arg(long)]
        include_models: bool,
    },
}

fn parse_family(s: &str) -> Option<ErrorFamily> {
    match s.to_lowercase().as_str() {
        "borrow_checker" | "borrow" => Some(ErrorFamily::BorrowChecker),
        "lifetimes" | "lifetime" => Some(ErrorFamily::Lifetimes),
        "trait_bounds" | "traits" => Some(ErrorFamily::TraitBounds),
        "result_handling" | "result" => Some(ErrorFamily::ResultHandling),
        "type_mismatch" | "types" => Some(ErrorFamily::TypeMismatch),
        "other" => Some(ErrorFamily::Other),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            count,
            family,
            max_attempts,
            model,
            run_id,
        } => {
            run_agent(count, family, max_attempts, model, run_id).await?;
        }
        Commands::Eval {
            cycle,
            max_attempts,
            model,
            save_episodes,
        } => {
            run_eval(cycle, max_attempts, model, &save_episodes).await?;
        }
        Commands::List { family, verbose } => {
            list_koans(family, verbose)?;
        }
        Commands::Show { id } => {
            show_koan(&id)?;
        }
        Commands::Export {
            output,
            failed_only,
            tasks,
        } => {
            export_data(&output, failed_only, tasks)?;
        }
        Commands::Sleep {
            episodes,
            tasks,
            output,
            quick,
            model,
        } => {
            run_sleep(episodes, tasks, &output, quick, &model)?;
        }
        Commands::Backup {
            output,
            include_models,
        } => {
            run_backup(&output, include_models)?;
        }
    }

    Ok(())
}

async fn run_agent(
    count: Option<usize>,
    family: Option<String>,
    max_attempts: u32,
    model: String,
    run_id: Option<String>,
) -> Result<()> {
    use agent::{AgentConfig, AgentLoop, OllamaClient, OllamaConfig};
    use tasks_rust_koans::{filter_by_family, get_random_koans, load_builtin_koans};

    println!("Loading koans...");
    let mut koans = load_builtin_koans();

    // Filter by family if specified
    if let Some(ref fam) = family {
        let error_family =
            parse_family(fam).with_context(|| format!("Unknown error family: {fam}"))?;
        koans = filter_by_family(&koans, error_family);
        println!(
            "Filtered to {} koans in {:?} family",
            koans.len(),
            error_family
        );
    }

    // Limit count if specified
    let koans = if let Some(n) = count {
        let family_filter = family.as_ref().and_then(|f| parse_family(f));
        get_random_koans(&koans, n.min(koans.len()), family_filter)
    } else {
        koans
    };

    println!("Running agent on {} koans...", koans.len());
    println!("Model: {model}");
    println!("Max attempts: {max_attempts}");
    println!();

    let llm_config = OllamaConfig {
        model: model.clone(),
        ..Default::default()
    };
    let llm = OllamaClient::new(llm_config);

    let agent_config = AgentConfig {
        max_attempts,
        run_id: run_id
            .unwrap_or_else(|| format!("run_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"))),
    };
    let agent = AgentLoop::new(llm, agent_config);

    let mut passed = 0;
    let mut failed = 0;

    for (i, task) in koans.iter().enumerate() {
        print!("[{}/{}] {} ... ", i + 1, koans.len(), task.id);

        match agent.run(task).await {
            Ok(result) => {
                if result.solved {
                    println!("PASS (attempts: {})", result.attempts);
                    passed += 1;
                } else {
                    println!("FAIL (max attempts reached)");
                    failed += 1;
                }
            }
            Err(e) => {
                println!("ERROR: {e}");
                failed += 1;
            }
        }
    }

    println!();
    println!(
        "Results: {passed}/{} passed ({:.1}%)",
        passed + failed,
        100.0 * passed as f64 / (passed + failed) as f64
    );

    Ok(())
}

async fn run_eval(cycle: u32, max_attempts: u32, model: String, save_dir: &str) -> Result<()> {
    use eval::{EvalConfig, EvalHarness};
    use std::fs;
    use std::io::Write;
    use std::path::Path;

    println!("Running evaluation on frozen eval set...");
    println!("Cycle: {cycle}");
    println!("Model: {model}");
    println!("Max attempts: {max_attempts}");
    println!();

    let config = EvalConfig {
        run_id: format!(
            "eval_cycle{cycle}_{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        ),
        cycle,
        max_attempts,
        model: model.clone(),
    };

    let harness = EvalHarness::new(config.clone());
    let run = harness
        .run_frozen_set()
        .await
        .context("Failed to run evaluation")?;

    // Save episodes to JSONL
    let save_path = Path::new(save_dir);
    fs::create_dir_all(save_path)?;
    let episodes_file = save_path.join(format!("cycle_{cycle}.jsonl"));
    let mut writer = std::io::BufWriter::new(fs::File::create(&episodes_file)?);
    for result in &run.task_results {
        serde_json::to_writer(&mut writer, &result.episode)?;
        writeln!(writer)?;
    }
    println!("Saved {} episodes to {}", run.task_count(), episodes_file.display());

    println!();
    println!("=== Evaluation Results ===");
    println!("Tasks evaluated: {}", run.task_count());
    println!("Pass rate: {:.1}%", run.pass_rate() * 100.0);

    if let Some(ref metrics) = run.metrics {
        println!("Passed: {}", metrics.passed);
        println!("Failed: {}", metrics.failed);
        println!(
            "Median steps to green: {:.1}",
            metrics.median_steps_to_green
        );

        if !metrics.error_signatures.is_empty() {
            println!();
            println!("Error signatures:");
            for (sig, count) in &metrics.error_signatures {
                println!("  {sig}: {count}");
            }
        }
    }

    // Save metrics summary
    let metrics_file = save_path.join("metrics.jsonl");
    let mut metrics_writer = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_file)?;
    if let Some(ref metrics) = run.metrics {
        let summary = serde_json::json!({
            "cycle": cycle,
            "run_id": config.run_id,
            "model": model,
            "pass_rate": metrics.pass_rate,
            "passed": metrics.passed,
            "failed": metrics.failed,
            "median_steps_to_green": metrics.median_steps_to_green,
            "error_signatures": metrics.error_signatures,
        });
        writeln!(metrics_writer, "{}", serde_json::to_string(&summary)?)?;
    }
    println!("Metrics appended to {}", metrics_file.display());

    Ok(())
}

fn list_koans(family: Option<String>, verbose: bool) -> Result<()> {
    use tasks_rust_koans::{filter_by_family, load_builtin_koans};

    let koans = load_builtin_koans();

    let koans = if let Some(ref fam) = family {
        let error_family =
            parse_family(fam).with_context(|| format!("Unknown error family: {fam}"))?;
        filter_by_family(&koans, error_family)
    } else {
        koans
    };

    println!("Available koans ({} total):", koans.len());
    println!();

    for koan in &koans {
        if verbose {
            println!("ID: {}", koan.id);
            println!("Family: {:?}", koan.family);
            println!("Description: {}", koan.description);
            println!("---");
        } else {
            println!("  {:?} | {} - {}", koan.family, koan.id, koan.description);
        }
    }

    Ok(())
}

fn show_koan(id: &str) -> Result<()> {
    use tasks_rust_koans::{get_koan_by_id, load_builtin_koans};

    let koans = load_builtin_koans();
    let koan = get_koan_by_id(&koans, id).with_context(|| format!("Koan not found: {id}"))?;

    println!("=== Koan: {} ===", koan.id);
    println!("Family: {:?}", koan.family);
    println!("Description: {}", koan.description);
    println!();
    println!("--- Buggy Code ---");
    println!("{}", koan.buggy_code);
    println!();
    println!("--- Correct Code ---");
    println!("{}", koan.correct_code);

    Ok(())
}

fn export_data(output_dir: &str, failed_only: bool, include_tasks: bool) -> Result<()> {
    use capture::EpisodeStore;
    use std::fs;
    use std::path::Path;
    use tasks_rust_koans::load_builtin_koans;

    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;

    // Export episodes from the store
    let store_path = Path::new("data/episodes.db");
    if store_path.exists() {
        let store = EpisodeStore::open(store_path)?;

        let episodes_path = if failed_only {
            let path = output_path.join("failed_episodes.jsonl");
            // Query failed episodes and export
            let failed = store.query_failed_then_fixed()?;
            println!("Found {} failed-then-fixed episodes", failed.len());

            // Write to JSONL
            let mut writer = std::io::BufWriter::new(fs::File::create(&path)?);
            for episode in &failed {
                use std::io::Write;
                serde_json::to_writer(&mut writer, episode)?;
                writeln!(writer)?;
            }
            path
        } else {
            let path = output_path.join("episodes.jsonl");
            let count = store.export_jsonl(&path)?;
            println!("Exported {count} episodes to {}", path.display());
            path
        };

        println!("Episodes saved to: {}", episodes_path.display());
    } else {
        println!("No episode store found at {}", store_path.display());
        println!("Run evaluations first to generate episodes.");
    }

    // Export tasks if requested
    if include_tasks {
        let koans = load_builtin_koans();
        let tasks_path = output_path.join("tasks.json");
        let file = fs::File::create(&tasks_path)?;
        serde_json::to_writer_pretty(file, &koans)?;
        println!("Exported {} tasks to {}", koans.len(), tasks_path.display());
    }

    Ok(())
}

fn run_sleep(
    episodes: Option<String>,
    tasks: Option<String>,
    output: &str,
    quick: bool,
    model: &str,
) -> Result<()> {
    use std::process::Command;

    println!("=== Sleepy Coder Training (Sleep Mode) ===");
    println!();

    // Step 1: Prepare data if not provided
    let episodes_path = episodes.unwrap_or_else(|| "data/export/episodes.jsonl".to_string());
    let tasks_path = tasks.unwrap_or_else(|| "data/export/tasks.json".to_string());

    // Check if data exists
    if !std::path::Path::new(&episodes_path).exists() {
        println!("No episodes found at {episodes_path}");
        println!("Run `sleepy-coder export` first to export training data.");
        return Ok(());
    }

    // Step 2: Prepare SFT dataset
    println!("Step 1: Preparing SFT dataset...");
    let sft_output = format!("{output}/sft_data.jsonl");
    let mut prepare_cmd = Command::new("sleepy-pact");
    prepare_cmd
        .args(["prepare", "-e", &episodes_path, "-o", &sft_output])
        .args(["-t", &tasks_path]);

    let status = prepare_cmd
        .status()
        .context("Failed to run sleepy-pact prepare. Is sleepy-pact installed?")?;

    if !status.success() {
        anyhow::bail!("sleepy-pact prepare failed");
    }
    println!("SFT dataset saved to {sft_output}");
    println!();

    // Step 3: Train LoRA adapter
    println!("Step 2: Training LoRA adapter...");
    let mut train_cmd = Command::new("sleepy-pact");
    train_cmd
        .args(["train", "-d", &sft_output, "-o", output, "-m", model]);

    if quick {
        train_cmd.arg("--quick");
    }

    let status = train_cmd
        .status()
        .context("Failed to run sleepy-pact train")?;

    if !status.success() {
        anyhow::bail!("sleepy-pact train failed");
    }

    println!();
    println!("=== Training Complete ===");
    println!("Adapter saved to: {output}");
    println!();
    println!("Next steps:");
    println!("  1. Run evaluation: sleepy-coder eval --cycle 1");
    println!("  2. Export for Ollama: sleepy-pact export-ollama -a {output}");

    Ok(())
}

fn run_backup(output_dir: &str, include_models: bool) -> Result<()> {
    use capture::EpisodeStore;
    use std::fs;
    use std::path::Path;

    let output_path = Path::new(output_dir);
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let backup_dir = output_path.join(format!("backup_{timestamp}"));
    fs::create_dir_all(&backup_dir)?;

    println!("=== Sleepy Coder Backup ===");
    println!("Backup directory: {}", backup_dir.display());
    println!();

    // 1. Backup episodes database to JSONL
    let db_path = Path::new("data/episodes.db");
    if db_path.exists() {
        let store = EpisodeStore::open(db_path)?;
        let episodes_path = backup_dir.join("episodes.jsonl");
        let count = store.export_jsonl(&episodes_path)?;
        println!("Exported {count} episodes to {}", episodes_path.display());
    } else {
        println!("No episodes database found");
    }

    // 2. Copy episode JSONL files from data/episodes/
    let episodes_dir = Path::new("data/episodes");
    if episodes_dir.exists() {
        let backup_episodes_dir = backup_dir.join("episodes");
        fs::create_dir_all(&backup_episodes_dir)?;

        let mut count = 0;
        for entry in fs::read_dir(episodes_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
                let dest = backup_episodes_dir.join(path.file_name().unwrap());
                fs::copy(&path, &dest)?;
                count += 1;
            }
        }
        if count > 0 {
            println!("Copied {count} episode files to {}", backup_episodes_dir.display());
        }
    }

    // 3. Copy metrics files
    let metrics_file = Path::new("data/episodes/metrics.jsonl");
    if metrics_file.exists() {
        let dest = backup_dir.join("metrics.jsonl");
        fs::copy(metrics_file, &dest)?;
        println!("Copied metrics to {}", dest.display());
    }

    // 4. Export tasks
    {
        use tasks_rust_koans::load_builtin_koans;
        let koans = load_builtin_koans();
        let tasks_path = backup_dir.join("tasks.json");
        let file = fs::File::create(&tasks_path)?;
        serde_json::to_writer_pretty(file, &koans)?;
        println!("Exported {} tasks to {}", koans.len(), tasks_path.display());
    }

    // 5. Optionally copy model checkpoints
    if include_models {
        let runs_dir = Path::new("runs");
        if runs_dir.exists() {
            let backup_models_dir = backup_dir.join("models");
            fs::create_dir_all(&backup_models_dir)?;

            // Copy safetensors files (safe checkpoint format)
            let mut copied = 0;
            for entry in walkdir::WalkDir::new(runs_dir)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
                    let relative = path.strip_prefix(runs_dir).unwrap();
                    let dest = backup_models_dir.join(relative);
                    if let Some(parent) = dest.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::copy(path, &dest)?;
                    copied += 1;
                }
            }
            if copied > 0 {
                println!("Copied {copied} model files to {}", backup_models_dir.display());
            }
        }
    }

    println!();
    println!("Backup complete!");
    println!();
    println!("To sync to Dropbox, copy this directory:");
    println!("  cp -r {} ~/Dropbox/sleepy-coder-backups/", backup_dir.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_family_valid() {
        assert_eq!(
            parse_family("borrow_checker"),
            Some(ErrorFamily::BorrowChecker)
        );
        assert_eq!(parse_family("borrow"), Some(ErrorFamily::BorrowChecker));
        assert_eq!(parse_family("lifetimes"), Some(ErrorFamily::Lifetimes));
        assert_eq!(parse_family("trait_bounds"), Some(ErrorFamily::TraitBounds));
        assert_eq!(
            parse_family("result_handling"),
            Some(ErrorFamily::ResultHandling)
        );
        assert_eq!(
            parse_family("type_mismatch"),
            Some(ErrorFamily::TypeMismatch)
        );
    }

    #[test]
    fn test_parse_family_case_insensitive() {
        assert_eq!(
            parse_family("BORROW_CHECKER"),
            Some(ErrorFamily::BorrowChecker)
        );
        assert_eq!(parse_family("Borrow"), Some(ErrorFamily::BorrowChecker));
    }

    #[test]
    fn test_parse_family_invalid() {
        assert_eq!(parse_family("unknown"), None);
        assert_eq!(parse_family(""), None);
    }

    #[test]
    fn test_cli_parse_list() {
        let cli = Cli::try_parse_from(["sleepy-coder", "list"]).unwrap();
        assert!(matches!(cli.command, Commands::List { .. }));
    }

    #[test]
    fn test_cli_parse_run_with_options() {
        let cli = Cli::try_parse_from([
            "sleepy-coder",
            "run",
            "--count",
            "5",
            "--family",
            "borrow",
            "--max-attempts",
            "3",
        ])
        .unwrap();

        match cli.command {
            Commands::Run {
                count,
                family,
                max_attempts,
                ..
            } => {
                assert_eq!(count, Some(5));
                assert_eq!(family, Some("borrow".to_string()));
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_parse_eval() {
        let cli = Cli::try_parse_from(["sleepy-coder", "eval", "--cycle", "5"]).unwrap();

        match cli.command {
            Commands::Eval { cycle, .. } => {
                assert_eq!(cycle, 5);
            }
            _ => panic!("Expected Eval command"),
        }
    }

    #[test]
    fn test_cli_parse_show() {
        let cli = Cli::try_parse_from(["sleepy-coder", "show", "borrow_001"]).unwrap();

        match cli.command {
            Commands::Show { id } => {
                assert_eq!(id, "borrow_001");
            }
            _ => panic!("Expected Show command"),
        }
    }
}
