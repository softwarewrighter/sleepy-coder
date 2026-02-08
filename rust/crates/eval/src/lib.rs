//! Evaluation harness for sleepy-coder.
//!
//! This crate provides:
//! - EvalHarness: Runs agent on a set of tasks and collects results
//! - EvalMetrics: Computes key metrics (repeat error rate, median steps, pass rate)
//! - EvalRun: Captures all data from a single evaluation run

use agent::{AgentConfig, AgentLoop, AgentResult, OllamaClient, OllamaConfig};
use chrono::{DateTime, Utc};
use core_types::{Episode, EvalResult, Task};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors from evaluation operations.
#[derive(Error, Debug)]
pub enum EvalError {
    #[error("Agent error: {0}")]
    Agent(#[from] agent::AgentError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("No tasks provided")]
    NoTasks,
}

/// Result type for evaluation operations.
pub type Result<T> = std::result::Result<T, EvalError>;

/// Configuration for an evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Run identifier
    pub run_id: String,
    /// Cycle number (for tracking progress over time)
    pub cycle: u32,
    /// Maximum attempts per task
    pub max_attempts: u32,
    /// Model to use
    pub model: String,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            run_id: format!("eval_{}", Utc::now().format("%Y%m%d_%H%M%S")),
            cycle: 0,
            max_attempts: 5,
            model: "qwen2.5-coder:1.5b-instruct-q4_K_M".to_string(),
        }
    }
}

/// Results from a single task evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvalResult {
    /// Task ID
    pub task_id: String,
    /// Whether the task was solved
    pub solved: bool,
    /// Number of attempts used
    pub attempts: u32,
    /// Error signature (if failed)
    pub error_signature: Option<String>,
    /// The episode generated
    pub episode: Episode,
}

impl From<AgentResult> for TaskEvalResult {
    fn from(result: AgentResult) -> Self {
        Self {
            task_id: result.task_id,
            solved: result.solved,
            attempts: result.attempts,
            error_signature: result.episode.error_signature.clone(),
            episode: result.episode,
        }
    }
}

/// Aggregated metrics from an evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// Total tasks evaluated
    pub total_tasks: u32,
    /// Tasks that passed
    pub passed: u32,
    /// Tasks that failed
    pub failed: u32,
    /// Pass rate (passed / total)
    pub pass_rate: f64,
    /// Median steps to green (for passing tasks)
    pub median_steps_to_green: f64,
    /// Error signature frequency map
    pub error_signatures: HashMap<String, u32>,
    /// Repeat error rate (compared to previous run)
    pub repeat_error_rate: f64,
}

impl EvalMetrics {
    /// Compute metrics from a list of task results.
    pub fn from_results(results: &[TaskEvalResult]) -> Self {
        let total_tasks = results.len() as u32;
        let passed = results.iter().filter(|r| r.solved).count() as u32;
        let failed = total_tasks - passed;
        let pass_rate = if total_tasks > 0 {
            passed as f64 / total_tasks as f64
        } else {
            0.0
        };

        // Compute median steps for passing tasks
        let mut steps: Vec<u32> = results
            .iter()
            .filter(|r| r.solved)
            .map(|r| r.attempts)
            .collect();
        steps.sort();

        let median_steps_to_green = if steps.is_empty() {
            0.0
        } else {
            let mid = steps.len() / 2;
            if steps.len() % 2 == 0 {
                (steps[mid - 1] + steps[mid]) as f64 / 2.0
            } else {
                steps[mid] as f64
            }
        };

        // Count error signatures
        let mut error_signatures = HashMap::new();
        for result in results {
            if let Some(ref sig) = result.error_signature {
                *error_signatures.entry(sig.clone()).or_insert(0) += 1;
            }
        }

        Self {
            total_tasks,
            passed,
            failed,
            pass_rate,
            median_steps_to_green,
            error_signatures,
            repeat_error_rate: 0.0, // Set later when comparing to previous run
        }
    }

    /// Compute repeat error rate by comparing to previous run's task-level error signatures.
    /// previous_task_errors maps task_id -> error_signature from the previous run.
    pub fn compute_repeat_rate(
        &mut self,
        current_task_errors: &HashMap<String, String>,
        previous_task_errors: &HashMap<String, String>,
    ) {
        if previous_task_errors.is_empty() {
            self.repeat_error_rate = 0.0;
            return;
        }

        let mut repeats = 0;
        let mut total = 0;

        for (task_id, current_sig) in current_task_errors {
            if let Some(prev_sig) = previous_task_errors.get(task_id) {
                total += 1;
                if prev_sig == current_sig {
                    repeats += 1;
                }
            }
        }

        self.repeat_error_rate = if total > 0 {
            repeats as f64 / total as f64
        } else {
            0.0
        };
    }

    /// Convert to EvalResult for storage.
    pub fn to_eval_result(&self, cycle: u32) -> EvalResult {
        let mut result = EvalResult::new(
            self.repeat_error_rate,
            self.median_steps_to_green,
            self.pass_rate,
            self.pass_rate, // frozen_pass_rate same as pass_rate for now
        );
        result.cycle = cycle;
        result
    }
}

/// A complete evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    /// Configuration used
    pub config: EvalConfig,
    /// When the run started
    pub started_at: DateTime<Utc>,
    /// When the run completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Individual task results
    pub task_results: Vec<TaskEvalResult>,
    /// Aggregated metrics
    pub metrics: Option<EvalMetrics>,
}

impl EvalRun {
    /// Create a new evaluation run.
    pub fn new(config: EvalConfig) -> Self {
        Self {
            config,
            started_at: Utc::now(),
            completed_at: None,
            task_results: Vec::new(),
            metrics: None,
        }
    }

    /// Add a task result.
    pub fn add_result(&mut self, result: TaskEvalResult) {
        self.task_results.push(result);
    }

    /// Finalize the run and compute metrics.
    pub fn finalize(&mut self) {
        self.completed_at = Some(Utc::now());
        self.metrics = Some(EvalMetrics::from_results(&self.task_results));
    }

    /// Get the number of tasks evaluated.
    pub fn task_count(&self) -> usize {
        self.task_results.len()
    }

    /// Get pass rate.
    pub fn pass_rate(&self) -> f64 {
        self.metrics.as_ref().map(|m| m.pass_rate).unwrap_or(0.0)
    }
}

/// The main evaluation harness.
pub struct EvalHarness {
    config: EvalConfig,
    llm_config: OllamaConfig,
}

impl EvalHarness {
    /// Create a new evaluation harness.
    pub fn new(config: EvalConfig) -> Self {
        let llm_config = OllamaConfig {
            model: config.model.clone(),
            ..Default::default()
        };

        Self { config, llm_config }
    }

    /// Create with custom LLM config.
    pub fn with_llm_config(config: EvalConfig, llm_config: OllamaConfig) -> Self {
        Self { config, llm_config }
    }

    /// Run evaluation on a set of tasks.
    pub async fn run(&self, tasks: &[Task]) -> Result<EvalRun> {
        if tasks.is_empty() {
            return Err(EvalError::NoTasks);
        }

        let mut eval_run = EvalRun::new(self.config.clone());

        let llm = OllamaClient::new(self.llm_config.clone());
        let agent_config = AgentConfig {
            max_attempts: self.config.max_attempts,
            run_id: self.config.run_id.clone(),
        };
        let agent = AgentLoop::new(llm, agent_config);

        for task in tasks {
            let result = agent.run(task).await?;
            eval_run.add_result(TaskEvalResult::from(result));
        }

        eval_run.finalize();
        Ok(eval_run)
    }

    /// Run evaluation on the frozen eval set.
    pub async fn run_frozen_set(&self) -> Result<EvalRun> {
        let tasks = tasks_rust_koans::get_frozen_eval_set();
        self.run(&tasks).await
    }
}

/// Compare two evaluation runs.
pub fn compare_runs(current: &EvalRun, previous: &EvalRun) -> RunComparison {
    let current_metrics = current.metrics.as_ref();
    let previous_metrics = previous.metrics.as_ref();

    let pass_rate_delta = match (current_metrics, previous_metrics) {
        (Some(c), Some(p)) => c.pass_rate - p.pass_rate,
        _ => 0.0,
    };

    let steps_delta = match (current_metrics, previous_metrics) {
        (Some(c), Some(p)) => c.median_steps_to_green - p.median_steps_to_green,
        _ => 0.0,
    };

    RunComparison {
        current_pass_rate: current_metrics.map(|m| m.pass_rate).unwrap_or(0.0),
        previous_pass_rate: previous_metrics.map(|m| m.pass_rate).unwrap_or(0.0),
        pass_rate_delta,
        current_median_steps: current_metrics
            .map(|m| m.median_steps_to_green)
            .unwrap_or(0.0),
        previous_median_steps: previous_metrics
            .map(|m| m.median_steps_to_green)
            .unwrap_or(0.0),
        steps_delta,
        improved: pass_rate_delta > 0.0 || (pass_rate_delta == 0.0 && steps_delta < 0.0),
    }
}

/// Comparison between two evaluation runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparison {
    /// Current run's pass rate
    pub current_pass_rate: f64,
    /// Previous run's pass rate
    pub previous_pass_rate: f64,
    /// Change in pass rate
    pub pass_rate_delta: f64,
    /// Current median steps
    pub current_median_steps: f64,
    /// Previous median steps
    pub previous_median_steps: f64,
    /// Change in median steps
    pub steps_delta: f64,
    /// Whether the current run is better
    pub improved: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::ErrorFamily;

    fn create_test_task(id: &str) -> Task {
        Task::new(
            id.to_string(),
            ErrorFamily::BorrowChecker,
            "Test task".to_string(),
            r#"fn main() { let s = String::new(); let t = s; println!("{}", s); }"#.to_string(),
            r#"fn main() { let s = String::new(); let t = s.clone(); println!("{}", s); }"#
                .to_string(),
        )
    }

    fn create_test_episode(task_id: &str, passed: bool, attempts: u32) -> Episode {
        let mut episode = Episode::new(task_id.to_string(), attempts, "test_run".to_string());
        if passed {
            episode.mark_passed(attempts);
        } else {
            episode.set_error("test_error".to_string());
        }
        episode
    }

    #[test]
    fn test_eval_config_default() {
        let config = EvalConfig::default();
        assert!(config.run_id.starts_with("eval_"));
        assert_eq!(config.cycle, 0);
        assert_eq!(config.max_attempts, 5);
    }

    #[test]
    fn test_task_eval_result_from_agent_result() {
        let episode = create_test_episode("task_001", true, 2);
        let agent_result = AgentResult {
            task_id: "task_001".to_string(),
            solved: true,
            attempts: 2,
            final_code: "fn main() {}".to_string(),
            attempt_history: vec![],
            episode,
        };

        let result: TaskEvalResult = agent_result.into();
        assert_eq!(result.task_id, "task_001");
        assert!(result.solved);
        assert_eq!(result.attempts, 2);
    }

    #[test]
    fn test_eval_metrics_from_results_all_pass() {
        let results = vec![
            TaskEvalResult {
                task_id: "t1".to_string(),
                solved: true,
                attempts: 1,
                error_signature: None,
                episode: create_test_episode("t1", true, 1),
            },
            TaskEvalResult {
                task_id: "t2".to_string(),
                solved: true,
                attempts: 3,
                error_signature: None,
                episode: create_test_episode("t2", true, 3),
            },
        ];

        let metrics = EvalMetrics::from_results(&results);
        assert_eq!(metrics.total_tasks, 2);
        assert_eq!(metrics.passed, 2);
        assert_eq!(metrics.failed, 0);
        assert!((metrics.pass_rate - 1.0).abs() < f64::EPSILON);
        // Median of [1, 3] = 2.0
        assert!((metrics.median_steps_to_green - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_metrics_from_results_mixed() {
        let results = vec![
            TaskEvalResult {
                task_id: "t1".to_string(),
                solved: true,
                attempts: 2,
                error_signature: None,
                episode: create_test_episode("t1", true, 2),
            },
            TaskEvalResult {
                task_id: "t2".to_string(),
                solved: false,
                attempts: 5,
                error_signature: Some("E0382:borrow_error".to_string()),
                episode: create_test_episode("t2", false, 5),
            },
        ];

        let metrics = EvalMetrics::from_results(&results);
        assert_eq!(metrics.total_tasks, 2);
        assert_eq!(metrics.passed, 1);
        assert_eq!(metrics.failed, 1);
        assert!((metrics.pass_rate - 0.5).abs() < f64::EPSILON);
        // Only 1 passing task with 2 steps
        assert!((metrics.median_steps_to_green - 2.0).abs() < f64::EPSILON);
        assert_eq!(metrics.error_signatures.get("E0382:borrow_error"), Some(&1));
    }

    #[test]
    fn test_eval_metrics_from_results_empty() {
        let results: Vec<TaskEvalResult> = vec![];
        let metrics = EvalMetrics::from_results(&results);

        assert_eq!(metrics.total_tasks, 0);
        assert_eq!(metrics.passed, 0);
        assert!((metrics.pass_rate - 0.0).abs() < f64::EPSILON);
        assert!((metrics.median_steps_to_green - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_metrics_to_eval_result() {
        let results = vec![TaskEvalResult {
            task_id: "t1".to_string(),
            solved: true,
            attempts: 2,
            error_signature: None,
            episode: create_test_episode("t1", true, 2),
        }];

        let metrics = EvalMetrics::from_results(&results);
        let eval_result = metrics.to_eval_result(3);

        assert_eq!(eval_result.cycle, 3);
        assert!((eval_result.pass_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_run_creation() {
        let config = EvalConfig::default();
        let run = EvalRun::new(config);

        assert!(run.task_results.is_empty());
        assert!(run.completed_at.is_none());
        assert!(run.metrics.is_none());
    }

    #[test]
    fn test_eval_run_add_result() {
        let config = EvalConfig::default();
        let mut run = EvalRun::new(config);

        let result = TaskEvalResult {
            task_id: "t1".to_string(),
            solved: true,
            attempts: 1,
            error_signature: None,
            episode: create_test_episode("t1", true, 1),
        };

        run.add_result(result);
        assert_eq!(run.task_count(), 1);
    }

    #[test]
    fn test_eval_run_finalize() {
        let config = EvalConfig::default();
        let mut run = EvalRun::new(config);

        run.add_result(TaskEvalResult {
            task_id: "t1".to_string(),
            solved: true,
            attempts: 1,
            error_signature: None,
            episode: create_test_episode("t1", true, 1),
        });

        run.finalize();

        assert!(run.completed_at.is_some());
        assert!(run.metrics.is_some());
        assert!((run.pass_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_harness_creation() {
        let config = EvalConfig::default();
        let harness = EvalHarness::new(config);

        assert_eq!(
            harness.llm_config.model,
            "qwen2.5-coder:1.5b-instruct-q4_K_M"
        );
    }

    #[test]
    fn test_compare_runs() {
        let config1 = EvalConfig {
            cycle: 1,
            ..Default::default()
        };
        let config2 = EvalConfig {
            cycle: 2,
            ..Default::default()
        };

        let mut run1 = EvalRun::new(config1);
        run1.add_result(TaskEvalResult {
            task_id: "t1".to_string(),
            solved: true,
            attempts: 3,
            error_signature: None,
            episode: create_test_episode("t1", true, 3),
        });
        run1.add_result(TaskEvalResult {
            task_id: "t2".to_string(),
            solved: false,
            attempts: 5,
            error_signature: Some("error".to_string()),
            episode: create_test_episode("t2", false, 5),
        });
        run1.finalize();

        let mut run2 = EvalRun::new(config2);
        run2.add_result(TaskEvalResult {
            task_id: "t1".to_string(),
            solved: true,
            attempts: 1,
            error_signature: None,
            episode: create_test_episode("t1", true, 1),
        });
        run2.add_result(TaskEvalResult {
            task_id: "t2".to_string(),
            solved: true,
            attempts: 2,
            error_signature: None,
            episode: create_test_episode("t2", true, 2),
        });
        run2.finalize();

        let comparison = compare_runs(&run2, &run1);

        // run2 has 100% pass rate, run1 has 50%
        assert!((comparison.current_pass_rate - 1.0).abs() < f64::EPSILON);
        assert!((comparison.previous_pass_rate - 0.5).abs() < f64::EPSILON);
        assert!(comparison.pass_rate_delta > 0.0);
        assert!(comparison.improved);
    }

    #[test]
    fn test_run_comparison_serialization() {
        let comparison = RunComparison {
            current_pass_rate: 0.8,
            previous_pass_rate: 0.6,
            pass_rate_delta: 0.2,
            current_median_steps: 2.0,
            previous_median_steps: 3.0,
            steps_delta: -1.0,
            improved: true,
        };

        let json = serde_json::to_string(&comparison).unwrap();
        let parsed: RunComparison = serde_json::from_str(&json).unwrap();

        assert!((parsed.pass_rate_delta - 0.2).abs() < f64::EPSILON);
        assert!(parsed.improved);
    }

    // Integration test requiring Ollama
    #[tokio::test]
    #[ignore]
    async fn test_eval_harness_run_integration() {
        let config = EvalConfig {
            run_id: "integration_test".to_string(),
            cycle: 0,
            max_attempts: 3,
            model: "qwen2.5-coder:1.5b-instruct-q4_K_M".to_string(),
        };

        let harness = EvalHarness::new(config);
        let tasks = vec![create_test_task("test_001")];

        let result = harness.run(&tasks).await.unwrap();
        println!("Pass rate: {:.2}%", result.pass_rate() * 100.0);
        println!("Tasks evaluated: {}", result.task_count());
    }
}
