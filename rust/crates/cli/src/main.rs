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
        } => {
            run_eval(cycle, max_attempts, model).await?;
        }
        Commands::List { family, verbose } => {
            list_koans(family, verbose)?;
        }
        Commands::Show { id } => {
            show_koan(&id)?;
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

async fn run_eval(cycle: u32, max_attempts: u32, model: String) -> Result<()> {
    use eval::{EvalConfig, EvalHarness};

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
        model,
    };

    let harness = EvalHarness::new(config);
    let run = harness
        .run_frozen_set()
        .await
        .context("Failed to run evaluation")?;

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
