//! Core types for sleepy-coder continual learning agent.
//!
//! This crate defines the shared data structures used across
//! the agent runtime, capture, training, and evaluation components.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema version for database migrations.
pub const SCHEMA_VERSION: u32 = 1;

/// Error family categories for clustering mistakes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorFamily {
    /// Borrow checker errors (moved values, borrows)
    BorrowChecker,
    /// Lifetime annotation issues
    Lifetimes,
    /// Missing trait implementations
    TraitBounds,
    /// Result/Option/? misuse
    ResultHandling,
    /// Iterator types, generics
    TypeMismatch,
    /// Other uncategorized errors
    Other,
}

/// A single episode of agent interaction with a task.
///
/// Captures the full trace of an attempt to solve a coding task,
/// including errors encountered and whether it succeeded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique task identifier
    pub task_id: String,
    /// Attempt number (0-indexed)
    pub attempt_idx: u32,
    /// Hash of the prompt used
    pub prompt_hash: String,
    /// Model identifier
    pub model_id: String,
    /// Normalized error signature (if any)
    pub error_signature: Option<String>,
    /// Unified diff of the fix
    pub diff_unified: Option<String>,
    /// Whether the task passed
    pub passed: bool,
    /// Number of tool calls until success (0 if failed)
    pub steps_to_green: u32,
    /// Wall clock time in milliseconds
    pub wall_clock_ms: u64,
    /// Input tokens used
    pub tokens_in: u32,
    /// Output tokens used
    pub tokens_out: u32,
    /// Timestamp of the episode
    pub timestamp: DateTime<Utc>,
}

impl Episode {
    /// Create a new episode with default values.
    pub fn new(task_id: String, attempt_idx: u32, model_id: String) -> Self {
        Self {
            task_id,
            attempt_idx,
            prompt_hash: String::new(),
            model_id,
            error_signature: None,
            diff_unified: None,
            passed: false,
            steps_to_green: 0,
            wall_clock_ms: 0,
            tokens_in: 0,
            tokens_out: 0,
            timestamp: Utc::now(),
        }
    }

    /// Mark the episode as passed with the number of steps taken.
    pub fn mark_passed(&mut self, steps: u32) {
        self.passed = true;
        self.steps_to_green = steps;
    }

    /// Set the normalized error signature.
    pub fn set_error(&mut self, signature: String) {
        self.error_signature = Some(signature);
    }

    /// Set the prompt hash from the actual prompt text.
    pub fn set_prompt(&mut self, prompt: &str) {
        let mut hasher = Sha256::new();
        hasher.update(prompt.as_bytes());
        self.prompt_hash = format!("{:x}", hasher.finalize());
    }
}

/// A coding task (koan) for the agent to solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: String,
    /// Error family for clustering
    pub family: ErrorFamily,
    /// Description of the task
    pub description: String,
    /// Buggy code to fix
    pub buggy_code: String,
    /// Correct solution
    pub correct_code: String,
    /// Expected error pattern (optional)
    pub expected_error: Option<String>,
}

impl Task {
    /// Create a new task.
    pub fn new(
        id: String,
        family: ErrorFamily,
        description: String,
        buggy_code: String,
        correct_code: String,
    ) -> Self {
        Self {
            id,
            family,
            description,
            buggy_code,
            correct_code,
            expected_error: None,
        }
    }
}

/// Evaluation results for a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Cycle number
    pub cycle: u32,
    /// Fraction of tasks with same error signature as before
    pub repeat_error_rate: f64,
    /// Median number of tool calls to success
    pub median_steps_to_green: f64,
    /// Pass rate on current tasks
    pub pass_rate: f64,
    /// Pass rate on frozen regression suite
    pub frozen_pass_rate: f64,
}

impl EvalResult {
    /// Create a new evaluation result.
    pub fn new(
        repeat_error_rate: f64,
        median_steps_to_green: f64,
        pass_rate: f64,
        frozen_pass_rate: f64,
    ) -> Self {
        Self {
            cycle: 0,
            repeat_error_rate,
            median_steps_to_green,
            pass_rate,
            frozen_pass_rate,
        }
    }
}

/// Normalize a raw compiler error into a stable signature.
///
/// Removes file paths, line numbers, and variable names to create
/// a consistent signature for clustering similar errors.
pub fn normalize_error(raw: &str) -> String {
    // Extract error code (e.g., E0382)
    let error_code = raw
        .find("error[E")
        .and_then(|start| {
            let rest = &raw[start + 6..];
            rest.find(']').map(|end| &rest[..end])
        })
        .unwrap_or("UNKNOWN");

    // Extract error type (text after the colon)
    let error_type = raw
        .find("]: ")
        .map(|pos| {
            let rest = &raw[pos + 3..];
            // Take until newline or backtick (variable name)
            rest.split('\n')
                .next()
                .unwrap_or("")
                .split('`')
                .next()
                .unwrap_or("")
                .trim()
                .replace(' ', "_")
                .to_lowercase()
        })
        .unwrap_or_else(|| "unknown".to_string());

    format!("E{error_code}:{error_type}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // RED: These tests define expected behavior - they should fail initially

    #[test]
    fn test_episode_creation() {
        let episode = Episode::new("task_001".to_string(), 0, "model_v1".to_string());

        assert_eq!(episode.task_id, "task_001");
        assert_eq!(episode.attempt_idx, 0);
        assert_eq!(episode.model_id, "model_v1");
        assert!(!episode.passed);
        assert_eq!(episode.steps_to_green, 0);
    }

    #[test]
    fn test_episode_mark_passed() {
        let mut episode = Episode::new("task_001".to_string(), 0, "model_v1".to_string());

        episode.mark_passed(3);

        assert!(episode.passed);
        assert_eq!(episode.steps_to_green, 3);
    }

    #[test]
    fn test_episode_set_error_signature() {
        let mut episode = Episode::new("task_001".to_string(), 0, "model_v1".to_string());

        episode.set_error("E0382:borrow_of_moved_value".to_string());

        assert_eq!(
            episode.error_signature,
            Some("E0382:borrow_of_moved_value".to_string())
        );
    }

    #[test]
    fn test_episode_serialization() {
        let episode = Episode::new("task_001".to_string(), 0, "model_v1".to_string());

        let json = serde_json::to_string(&episode).unwrap();
        let parsed: Episode = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.task_id, episode.task_id);
    }

    #[test]
    fn test_task_creation() {
        let task = Task::new(
            "task_001".to_string(),
            ErrorFamily::BorrowChecker,
            "Fix the moved value error".to_string(),
            "fn main() { let s = String::new(); let t = s; println!(\"{}\", s); }".to_string(),
            "fn main() { let s = String::new(); let t = s.clone(); println!(\"{}\", s); }"
                .to_string(),
        );

        assert_eq!(task.id, "task_001");
        assert_eq!(task.family, ErrorFamily::BorrowChecker);
    }

    #[test]
    fn test_task_serialization() {
        let task = Task::new(
            "task_001".to_string(),
            ErrorFamily::BorrowChecker,
            "Fix error".to_string(),
            "buggy".to_string(),
            "correct".to_string(),
        );

        let json = serde_json::to_string(&task).unwrap();
        let parsed: Task = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, task.id);
        assert_eq!(parsed.family, task.family);
    }

    #[test]
    fn test_error_family_variants() {
        let families = vec![
            ErrorFamily::BorrowChecker,
            ErrorFamily::Lifetimes,
            ErrorFamily::TraitBounds,
            ErrorFamily::ResultHandling,
            ErrorFamily::TypeMismatch,
            ErrorFamily::Other,
        ];

        // All variants should serialize/deserialize
        for family in families {
            let json = serde_json::to_string(&family).unwrap();
            let parsed: ErrorFamily = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, family);
        }
    }

    #[test]
    fn test_eval_result_creation() {
        let result = EvalResult::new(
            0.25, // repeat_error_rate
            3.5,  // median_steps_to_green
            0.85, // pass_rate
            0.90, // frozen_pass_rate
        );

        assert!((result.repeat_error_rate - 0.25).abs() < f64::EPSILON);
        assert!((result.median_steps_to_green - 3.5).abs() < f64::EPSILON);
        assert!((result.pass_rate - 0.85).abs() < f64::EPSILON);
        assert!((result.frozen_pass_rate - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_result_serialization() {
        let result = EvalResult::new(0.25, 3.5, 0.85, 0.90);

        let json = serde_json::to_string(&result).unwrap();
        let parsed: EvalResult = serde_json::from_str(&json).unwrap();

        assert!((parsed.repeat_error_rate - result.repeat_error_rate).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_error_removes_paths() {
        let raw = r#"error[E0382]: borrow of moved value: `s`
 --> src/main.rs:3:20
  |
2 |     let t = s;
  |             - value moved here
3 |     println!("{}", s);
  |                    ^ value borrowed here after move"#;

        let normalized = normalize_error(raw);

        // Should contain error code and type
        assert!(normalized.contains("E0382"));
        assert!(normalized.contains("borrow_of_moved_value"));

        // Should NOT contain file paths
        assert!(!normalized.contains("src/main.rs"));
        assert!(!normalized.contains("-->"));
    }

    #[test]
    fn test_normalize_error_handles_variable_names() {
        let raw1 = "error[E0382]: borrow of moved value: `foo`";
        let raw2 = "error[E0382]: borrow of moved value: `bar`";

        let norm1 = normalize_error(raw1);
        let norm2 = normalize_error(raw2);

        // Same error type should normalize to same signature
        assert_eq!(norm1, norm2);
    }

    #[test]
    fn test_schema_version() {
        // Schema version should be defined for migrations
        let version = SCHEMA_VERSION;
        assert!(version > 0, "Schema version must be positive");
        assert_eq!(version, 1, "Current schema version");
    }
}
