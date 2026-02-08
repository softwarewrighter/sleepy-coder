//! Sandboxed cargo check/test execution for sleepy-coder.
//!
//! This crate provides isolated execution of Rust code in temporary
//! directories, capturing compiler output and test results.

use core_types::Task;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;
use thiserror::Error;

/// Errors from sandbox operations.
#[derive(Error, Debug)]
pub enum SandboxError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to create sandbox directory")]
    DirectoryCreation,

    #[error("Failed to run cargo: {0}")]
    CargoExecution(String),
}

/// Result type for sandbox operations.
pub type Result<T> = std::result::Result<T, SandboxError>;

/// A parsed compiler error.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Error code (e.g., "E0382")
    pub code: String,
    /// Error message
    pub message: String,
    /// Line number (if available)
    pub line: Option<u32>,
    /// Column number (if available)
    pub column: Option<u32>,
}

impl CompileError {
    /// Create a normalized error signature for clustering.
    pub fn normalized_signature(&self) -> String {
        // Remove variable names (backtick-quoted) and normalize
        let msg = self
            .message
            .split('`')
            .next()
            .unwrap_or(&self.message)
            .trim()
            .trim_end_matches(':')
            .trim()
            .replace(' ', "_")
            .to_lowercase();

        format!("{}:{msg}", self.code)
    }
}

/// Result of running cargo check.
#[derive(Debug)]
pub struct CompileResult {
    /// Whether compilation succeeded
    pub success: bool,
    /// Parsed errors
    pub errors: Vec<CompileError>,
    /// Raw stdout
    pub stdout: String,
    /// Raw stderr
    pub stderr: String,
}

impl CompileResult {
    /// Parse compile result from cargo output.
    pub fn from_output(success: bool, stdout: &str, stderr: &str) -> Self {
        let errors = Self::parse_errors(stderr);
        Self {
            success,
            errors,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
        }
    }

    /// Parse error codes and messages from stderr.
    fn parse_errors(stderr: &str) -> Vec<CompileError> {
        let mut errors = Vec::new();

        for line in stderr.lines() {
            // Match lines like: error[E0382]: borrow of moved value: `s`
            if line.starts_with("error[E")
                && let Some(bracket_end) = line.find("]: ")
            {
                let code_start = 6; // "error[" length
                let code = &line[code_start..bracket_end];
                let message = &line[bracket_end + 3..];

                errors.push(CompileError {
                    code: code.to_string(),
                    message: message.to_string(),
                    line: None,
                    column: None,
                });
            }
        }

        errors
    }
}

/// Result of running cargo test.
#[derive(Debug)]
pub struct TestResult {
    /// Whether all tests passed
    pub success: bool,
    /// Number of tests passed
    pub tests_passed: u32,
    /// Number of tests failed
    pub tests_failed: u32,
    /// Raw stdout
    pub stdout: String,
    /// Raw stderr
    pub stderr: String,
}

impl TestResult {
    /// Parse test result from cargo test output.
    pub fn from_output(success: bool, stdout: &str, stderr: &str) -> Self {
        let (passed, failed) = Self::parse_test_counts(stdout);
        Self {
            success,
            tests_passed: passed,
            tests_failed: failed,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
        }
    }

    /// Parse test counts from stdout.
    fn parse_test_counts(stdout: &str) -> (u32, u32) {
        // Look for line like: "test result: ok. 1 passed; 0 failed"
        // or "test result: FAILED. 0 passed; 1 failed"
        for line in stdout.lines() {
            if line.starts_with("test result:") {
                let mut passed = 0u32;
                let mut failed = 0u32;

                // Parse "X passed"
                if let Some(pos) = line.find(" passed") {
                    let before = &line[..pos];
                    if let Some(num_start) = before.rfind(|c: char| !c.is_ascii_digit())
                        && let Ok(n) = before[num_start + 1..].parse()
                    {
                        passed = n;
                    }
                }

                // Parse "X failed"
                if let Some(pos) = line.find(" failed") {
                    let before = &line[..pos];
                    if let Some(num_start) = before.rfind(|c: char| !c.is_ascii_digit())
                        && let Ok(n) = before[num_start + 1..].parse()
                    {
                        failed = n;
                    }
                }

                return (passed, failed);
            }
        }

        (0, 0)
    }
}

/// An isolated sandbox for running Rust code.
pub struct Sandbox {
    /// Temporary directory (cleaned up on drop)
    temp_dir: TempDir,
    /// Path to main.rs
    main_rs_path: PathBuf,
}

impl Sandbox {
    /// Create a new sandbox from a task.
    pub fn new(task: &Task) -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let sandbox_path = temp_dir.path();

        // Create src directory
        let src_dir = sandbox_path.join("src");
        fs::create_dir(&src_dir)?;

        // Write Cargo.toml
        let cargo_toml = format!(
            r#"[package]
name = "sandbox_{}"
version = "0.1.0"
edition = "2021"

[dependencies]
"#,
            task.id.replace('-', "_")
        );
        fs::write(sandbox_path.join("Cargo.toml"), cargo_toml)?;

        // Write main.rs with the buggy code
        let main_rs_path = src_dir.join("main.rs");
        fs::write(&main_rs_path, &task.buggy_code)?;

        Ok(Self {
            temp_dir,
            main_rs_path,
        })
    }

    /// Get the path to the sandbox directory.
    pub fn path(&self) -> &Path {
        self.temp_dir.path()
    }

    /// Update the code in main.rs.
    pub fn update_code(&mut self, code: &str) -> Result<()> {
        fs::write(&self.main_rs_path, code)?;
        Ok(())
    }

    /// Run cargo check and return the result.
    pub fn run_check(&self) -> Result<CompileResult> {
        let output = Command::new("cargo")
            .args(["check"])
            .current_dir(self.path())
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(CompileResult::from_output(
            output.status.success(),
            &stdout,
            &stderr,
        ))
    }

    /// Run cargo test and return the result.
    pub fn run_test(&self) -> Result<TestResult> {
        let output = Command::new("cargo")
            .args(["test"])
            .current_dir(self.path())
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(TestResult::from_output(
            output.status.success(),
            &stdout,
            &stderr,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::{ErrorFamily, Task};

    #[test]
    fn test_sandbox_creation() {
        let task = create_test_task();
        let sandbox = Sandbox::new(&task).unwrap();

        // Sandbox should create temp directory with Cargo.toml and src/main.rs
        assert!(sandbox.path().exists());
        assert!(sandbox.path().join("Cargo.toml").exists());
        assert!(sandbox.path().join("src/main.rs").exists());
    }

    #[test]
    fn test_sandbox_runs_cargo_check() {
        let task = Task::new(
            "valid_task".to_string(),
            ErrorFamily::Other,
            "Valid code".to_string(),
            r#"fn main() { println!("Hello"); }"#.to_string(),
            r#"fn main() { println!("Hello"); }"#.to_string(),
        );

        let sandbox = Sandbox::new(&task).unwrap();
        let result = sandbox.run_check().unwrap();

        assert!(result.success);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_sandbox_captures_compile_error() {
        let task = Task::new(
            "borrow_error".to_string(),
            ErrorFamily::BorrowChecker,
            "Borrow checker error".to_string(),
            r#"fn main() { let s = String::new(); let t = s; println!("{}", s); }"#.to_string(),
            r#"fn main() { let s = String::new(); let t = s.clone(); println!("{}", s); }"#
                .to_string(),
        );

        let sandbox = Sandbox::new(&task).unwrap();
        let result = sandbox.run_check().unwrap();

        assert!(!result.success);
        assert!(!result.errors.is_empty());
        assert!(result.stderr.contains("E0382") || result.stderr.contains("borrow"));
    }

    #[test]
    fn test_sandbox_apply_patch() {
        let task = Task::new(
            "patch_test".to_string(),
            ErrorFamily::BorrowChecker,
            "Test patching".to_string(),
            r#"fn main() { let s = String::new(); let t = s; println!("{}", s); }"#.to_string(),
            r#"fn main() { let s = String::new(); let t = s.clone(); println!("{}", s); }"#
                .to_string(),
        );

        let mut sandbox = Sandbox::new(&task).unwrap();

        // First check should fail
        let result1 = sandbox.run_check().unwrap();
        assert!(!result1.success);

        // Apply fix
        let new_code =
            r#"fn main() { let s = String::new(); let t = s.clone(); println!("{}", s); }"#;
        sandbox.update_code(new_code).unwrap();

        // Second check should pass
        let result2 = sandbox.run_check().unwrap();
        assert!(result2.success);
    }

    #[test]
    fn test_sandbox_runs_cargo_test() {
        let task = Task::new(
            "test_task".to_string(),
            ErrorFamily::Other,
            "Code with tests".to_string(),
            r#"
fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}

fn main() {}
"#
            .to_string(),
            "same".to_string(),
        );

        let sandbox = Sandbox::new(&task).unwrap();
        let result = sandbox.run_test().unwrap();

        assert!(result.success);
        assert!(result.tests_passed > 0);
    }

    #[test]
    fn test_sandbox_captures_test_failure() {
        let task = Task::new(
            "failing_test".to_string(),
            ErrorFamily::Other,
            "Code with failing test".to_string(),
            r#"
fn add(a: i32, b: i32) -> i32 { a + b + 1 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}

fn main() {}
"#
            .to_string(),
            "same".to_string(),
        );

        let sandbox = Sandbox::new(&task).unwrap();
        let result = sandbox.run_test().unwrap();

        assert!(!result.success);
        assert!(result.tests_failed > 0);
    }

    #[test]
    fn test_compile_result_error_parsing() {
        let stderr = r#"error[E0382]: borrow of moved value: `s`
 --> src/main.rs:1:52
  |
1 | fn main() { let s = String::new(); let t = s; println!("{}", s); }
  |                 -                          -               ^ value borrowed here after move
  |                 |                          |
  |                 |                          value moved here
  |                 move occurs because `s` has type `String`, which does not implement the `Copy` trait

error: aborting due to 1 previous error
"#;

        let result = CompileResult::from_output(false, "", stderr);

        assert!(!result.success);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "E0382");
        assert!(result.errors[0].message.contains("borrow of moved value"));
    }

    #[test]
    fn test_normalize_error_signature() {
        let error = CompileError {
            code: "E0382".to_string(),
            message: "borrow of moved value: `foo`".to_string(),
            line: Some(1),
            column: Some(52),
        };

        let sig = error.normalized_signature();
        assert_eq!(sig, "E0382:borrow_of_moved_value");
    }

    #[test]
    fn test_sandbox_cleanup() {
        let task = create_test_task();
        let path;
        {
            let sandbox = Sandbox::new(&task).unwrap();
            path = sandbox.path().to_path_buf();
            assert!(path.exists());
        }
        // Sandbox should clean up on drop
        assert!(!path.exists());
    }

    fn create_test_task() -> Task {
        Task::new(
            "test".to_string(),
            ErrorFamily::Other,
            "Test task".to_string(),
            r#"fn main() { println!("test"); }"#.to_string(),
            r#"fn main() { println!("test"); }"#.to_string(),
        )
    }
}
