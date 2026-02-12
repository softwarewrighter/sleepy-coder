//! Minimal Pi-like agent loop for sleepy-coder.
//!
//! This crate provides:
//! - OllamaClient: API client for local LLM inference via Ollama
//! - AgentLoop: The RED→patch→GREEN loop for fixing buggy code

use core_types::{Episode, Task};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from agent operations.
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("LLM returned empty response")]
    EmptyResponse,

    #[error("Sandbox error: {0}")]
    Sandbox(#[from] sandbox::SandboxError),

    #[error("Max attempts exceeded: {0}")]
    MaxAttemptsExceeded(u32),
}

/// Result type for agent operations.
pub type Result<T> = std::result::Result<T, AgentError>;

/// Configuration for the Ollama client.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    /// Base URL (default: http://localhost:11434)
    pub base_url: String,
    /// Model name (default: qwen2.5-coder:1.5b-instruct-q4_K_M)
    pub model: String,
    /// Temperature for generation (default: 0.2)
    pub temperature: f32,
    /// Maximum tokens to generate (default: 2048)
    pub max_tokens: u32,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "qwen2.5-coder:1.5b-instruct-q4_K_M".to_string(),
            temperature: 0.2,
            max_tokens: 2048,
        }
    }
}

/// Request body for Ollama /api/generate.
#[derive(Debug, Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: OllamaOptions,
}

/// Options for Ollama generation.
#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: u32,
}

/// Response from Ollama /api/generate.
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    /// Generated text
    pub response: String,
    /// Whether generation is complete
    pub done: bool,
    /// Total duration in nanoseconds
    #[serde(default)]
    pub total_duration: u64,
    /// Tokens generated
    #[serde(default)]
    pub eval_count: u32,
}

/// Client for Ollama API.
pub struct OllamaClient {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Create a new OllamaClient with the given config.
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Generate a completion from the LLM.
    pub async fn generate(&self, prompt: &str) -> Result<OllamaResponse> {
        let url = format!("{}/api/generate", self.config.base_url);

        let request = OllamaRequest {
            model: &self.config.model,
            prompt,
            stream: false,
            options: OllamaOptions {
                temperature: self.config.temperature,
                num_predict: self.config.max_tokens,
            },
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json::<OllamaResponse>()
            .await?;

        if response.response.is_empty() {
            return Err(AgentError::EmptyResponse);
        }

        Ok(response)
    }

    /// Generate a code fix for the given buggy code and error message.
    pub async fn generate_fix(&self, buggy_code: &str, error_message: &str) -> Result<String> {
        let hints = Self::get_error_hints(error_message);
        let prompt = format!(
            r#"Fix the following Rust code that has a compilation error.

## Buggy Code:
```rust
{buggy_code}
```

## Compiler Error:
{error_message}
{hints}
## Instructions:
- Return ONLY the fixed Rust code
- Do not include any explanation
- Do not include markdown code fences
- The code should compile without errors

## Fixed Code:
"#
        );

        let response = self.generate(&prompt).await?;
        Ok(Self::extract_code(&response.response))
    }

    /// Get error-specific hints based on the compiler error message.
    fn get_error_hints(error_message: &str) -> String {
        let error_lower = error_message.to_lowercase();

        // Borrow checker hints
        if error_lower.contains("cannot borrow")
            && error_lower.contains("mutable")
            && error_lower.contains("immutable")
        {
            return r#"
## Hint (Mutable/Immutable Borrow Conflict):
When you have an immutable borrow (&T), you cannot create a mutable borrow (&mut T) while the immutable borrow is still in use.
FIX: Either copy the value instead of borrowing (remove &), or restructure code so borrows don't overlap.
"#
            .to_string();
        }

        if error_lower.contains("cannot borrow") && error_lower.contains("mutable more than once") {
            return r#"
## Hint (Double Mutable Borrow):
Rust allows only ONE mutable reference at a time. Two &mut references cannot coexist.
FIX: Use the first mutable borrow completely before creating the second one. Restructure so borrows are sequential, not overlapping.
"#
            .to_string();
        }

        if error_lower.contains("returns a reference to data owned by the current function")
            || error_lower.contains("cannot return reference to local variable")
            || (error_lower.contains("borrowed value does not live long enough")
                && error_lower.contains("return"))
        {
            return r#"
## Hint (Returning Reference to Local):
You cannot return a reference (&T or &str) to a value created inside the function - it will be dropped.
FIX: Return an owned type instead. Change &str to String, &T to T, or use .to_string()/.to_owned()/.clone().
"#
            .to_string();
        }

        // Result/Option handling hints
        if error_lower.contains("expected")
            && error_lower.contains("result")
            && error_lower.contains("option")
        {
            return r#"
## Hint (Option to Result Conversion):
Option<T> and Result<T, E> are different types. You need to convert between them.
FIX: Use .ok_or("error message") or .ok_or_else(|| Error) to convert Option to Result.
"#
            .to_string();
        }

        if error_lower.contains("the `?` operator can only be used")
            || error_lower.contains("cannot use the `?` operator")
        {
            return r#"
## Hint (? Operator Requires Result Return):
The ? operator can only be used in functions that return Result or Option.
FIX: Change the function signature to return Result<T, E>. For main(), use:
fn main() -> Result<(), Box<dyn std::error::Error>> { ... Ok(()) }
"#
            .to_string();
        }

        // Trait bounds hints - check for both "doesn't implement" and "is not implemented"
        let missing_trait = error_lower.contains("doesn't implement")
            || error_lower.contains("is not implemented");

        if missing_trait && error_lower.contains("clone") {
            return r#"
## Hint (Missing Clone):
The Clone trait is not automatically implemented. You must derive or implement it.
FIX: Add #[derive(Clone)] above the struct definition.
"#
            .to_string();
        }

        if missing_trait && (error_lower.contains("hash") || error_lower.contains("hashmap")) {
            return r#"
## Hint (Missing Hash for HashMap Key):
HashMap keys must implement Hash, PartialEq, and Eq traits.
FIX: Add #[derive(Hash, PartialEq, Eq)] above the struct definition. All three are required together.
"#
            .to_string();
        }

        if missing_trait && error_lower.contains("ord") {
            return r#"
## Hint (Missing Ord for Sorting):
Sorting requires Ord trait, which depends on PartialOrd, PartialEq, and Eq.
FIX: Add #[derive(PartialEq, Eq, PartialOrd, Ord)] above the struct definition. All four are required.
"#
            .to_string();
        }

        // Generic trait bound hint
        if error_lower.contains("the trait bound") && error_lower.contains("is not satisfied") {
            return r#"
## Hint (Missing Trait Bound):
A required trait is not implemented for this type.
FIX: Add #[derive(...)] with the missing trait, or add a trait bound to the generic parameter.
"#
            .to_string();
        }

        // No specific hint found
        String::new()
    }

    /// Extract code from LLM response, stripping markdown fences if present.
    fn extract_code(response: &str) -> String {
        let trimmed = response.trim();

        // If wrapped in ```rust ... ```, extract the content
        if trimmed.starts_with("```rust")
            && let Some(end) = trimmed.rfind("```")
        {
            let start = "```rust".len();
            if end > start {
                return trimmed[start..end].trim().to_string();
            }
        }

        // If wrapped in ``` ... ```, extract the content
        if trimmed.starts_with("```")
            && let Some(end) = trimmed.rfind("```")
        {
            let start = 3; // "```".len()
            if end > start {
                return trimmed[start..end].trim().to_string();
            }
        }

        trimmed.to_string()
    }
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum fix attempts before giving up
    pub max_attempts: u32,
    /// Run ID for episode tracking
    pub run_id: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            run_id: "default".to_string(),
        }
    }
}

/// Result of a single fix attempt.
#[derive(Debug)]
pub struct AttemptResult {
    /// The code that was tried
    pub code: String,
    /// Whether compilation succeeded
    pub success: bool,
    /// Compiler output (errors or success message)
    pub compiler_output: String,
    /// Attempt number (1-indexed)
    pub attempt: u32,
}

/// Result of running the agent loop on a task.
#[derive(Debug)]
pub struct AgentResult {
    /// The task that was processed
    pub task_id: String,
    /// Whether the task was solved
    pub solved: bool,
    /// Number of attempts made
    pub attempts: u32,
    /// Final code (fixed or last attempt)
    pub final_code: String,
    /// History of all attempts
    pub attempt_history: Vec<AttemptResult>,
    /// Generated episode for capture
    pub episode: Episode,
}

/// The main agent loop for fixing buggy code.
pub struct AgentLoop {
    llm: OllamaClient,
    config: AgentConfig,
}

impl AgentLoop {
    /// Create a new AgentLoop with the given LLM client and config.
    pub fn new(llm: OllamaClient, config: AgentConfig) -> Self {
        Self { llm, config }
    }

    /// Run the agent on a task, attempting to fix the buggy code.
    pub async fn run(&self, task: &Task) -> Result<AgentResult> {
        use sandbox::Sandbox;

        let mut sandbox = Sandbox::new(task)?;
        let mut attempts = Vec::new();
        let mut current_code = task.buggy_code.clone();

        for attempt_num in 1..=self.config.max_attempts {
            // Check if current code compiles
            let check_result = sandbox.run_check()?;

            let attempt = AttemptResult {
                code: current_code.clone(),
                success: check_result.success,
                compiler_output: check_result.stderr.clone(),
                attempt: attempt_num,
            };
            attempts.push(attempt);

            if check_result.success {
                // Code compiles! Create success episode
                let mut episode =
                    Episode::new(task.id.clone(), attempt_num, self.config.run_id.clone());
                episode.mark_passed(attempt_num);

                return Ok(AgentResult {
                    task_id: task.id.clone(),
                    solved: true,
                    attempts: attempt_num,
                    final_code: current_code,
                    attempt_history: attempts,
                    episode,
                });
            }

            // Generate a fix
            let fixed_code = self
                .llm
                .generate_fix(&current_code, &check_result.stderr)
                .await?;

            // Update sandbox with new code
            current_code = fixed_code;
            sandbox.update_code(&current_code)?;
        }

        // Max attempts exceeded - create failed episode
        let mut episode = Episode::new(
            task.id.clone(),
            self.config.max_attempts,
            self.config.run_id.clone(),
        );
        episode.set_error("max_attempts_exceeded".to_string());

        Ok(AgentResult {
            task_id: task.id.clone(),
            solved: false,
            attempts: self.config.max_attempts,
            final_code: current_code,
            attempt_history: attempts,
            episode,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::ErrorFamily;

    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "qwen2.5-coder:1.5b-instruct-q4_K_M");
        assert!((config.temperature - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens, 2048);
    }

    #[test]
    fn test_ollama_config_custom() {
        let config = OllamaConfig {
            base_url: "http://gpu-server:11434".to_string(),
            model: "codellama:7b".to_string(),
            temperature: 0.5,
            max_tokens: 4096,
        };
        assert_eq!(config.base_url, "http://gpu-server:11434");
        assert_eq!(config.model, "codellama:7b");
    }

    #[test]
    fn test_extract_code_plain() {
        let response = "fn main() { println!(\"hello\"); }";
        let extracted = OllamaClient::extract_code(response);
        assert_eq!(extracted, "fn main() { println!(\"hello\"); }");
    }

    #[test]
    fn test_extract_code_rust_fence() {
        let response = "```rust\nfn main() { println!(\"hello\"); }\n```";
        let extracted = OllamaClient::extract_code(response);
        assert_eq!(extracted, "fn main() { println!(\"hello\"); }");
    }

    #[test]
    fn test_extract_code_plain_fence() {
        let response = "```\nfn main() { println!(\"hello\"); }\n```";
        let extracted = OllamaClient::extract_code(response);
        assert_eq!(extracted, "fn main() { println!(\"hello\"); }");
    }

    #[test]
    fn test_extract_code_with_whitespace() {
        let response = "  \n```rust\nfn main() {}\n```  \n";
        let extracted = OllamaClient::extract_code(response);
        assert_eq!(extracted, "fn main() {}");
    }

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_attempts, 5);
        assert_eq!(config.run_id, "default");
    }

    #[test]
    fn test_attempt_result_structure() {
        let result = AttemptResult {
            code: "fn main() {}".to_string(),
            success: true,
            compiler_output: "".to_string(),
            attempt: 1,
        };
        assert!(result.success);
        assert_eq!(result.attempt, 1);
    }

    #[test]
    fn test_ollama_request_serialization() {
        let request = OllamaRequest {
            model: "test-model",
            prompt: "test prompt",
            stream: false,
            options: OllamaOptions {
                temperature: 0.2,
                num_predict: 1024,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"test-model\""));
        assert!(json.contains("\"stream\":false"));
        assert!(json.contains("\"temperature\":0.2"));
    }

    #[test]
    fn test_ollama_response_deserialization() {
        let json = r#"{
            "response": "fn main() {}",
            "done": true,
            "total_duration": 1000000,
            "eval_count": 10
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.response, "fn main() {}");
        assert!(response.done);
        assert_eq!(response.total_duration, 1000000);
        assert_eq!(response.eval_count, 10);
    }

    #[test]
    fn test_ollama_response_minimal() {
        // Ollama sometimes returns minimal responses
        let json = r#"{"response": "hello", "done": true}"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.response, "hello");
        assert!(response.done);
        assert_eq!(response.total_duration, 0); // default
        assert_eq!(response.eval_count, 0); // default
    }

    #[tokio::test]
    async fn test_ollama_client_creation() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);
        assert_eq!(client.config.model, "qwen2.5-coder:1.5b-instruct-q4_K_M");
    }

    #[test]
    fn test_agent_loop_creation() {
        let llm = OllamaClient::new(OllamaConfig::default());
        let config = AgentConfig {
            max_attempts: 3,
            run_id: "test-run".to_string(),
        };
        let agent = AgentLoop::new(llm, config);
        assert_eq!(agent.config.max_attempts, 3);
        assert_eq!(agent.config.run_id, "test-run");
    }

    // Integration test that requires Ollama running - skip in CI
    #[tokio::test]
    #[ignore]
    async fn test_ollama_generate_integration() {
        let config = OllamaConfig::default();
        let client = OllamaClient::new(config);

        let response = client.generate("Say hello in Rust code").await.unwrap();
        assert!(!response.response.is_empty());
        assert!(response.done);
    }

    // Integration test for the full agent loop
    #[tokio::test]
    #[ignore]
    async fn test_agent_loop_integration() {
        let llm = OllamaClient::new(OllamaConfig::default());
        let config = AgentConfig {
            max_attempts: 5,
            run_id: "integration-test".to_string(),
        };
        let agent = AgentLoop::new(llm, config);

        let task = Task::new(
            "test_borrow".to_string(),
            ErrorFamily::BorrowChecker,
            "Fix borrow error".to_string(),
            r#"fn main() { let s = String::new(); let t = s; println!("{}", s); }"#.to_string(),
            r#"fn main() { let s = String::new(); let t = s.clone(); println!("{}", s); }"#
                .to_string(),
        );

        let result = agent.run(&task).await.unwrap();
        println!("Solved: {}, Attempts: {}", result.solved, result.attempts);
        println!("Final code: {}", result.final_code);
    }

    // Tests for error hints
    #[test]
    fn test_hint_mutable_immutable_conflict() {
        let error = "cannot borrow `v` as mutable because it is also borrowed as immutable";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Mutable/Immutable Borrow Conflict"));
        assert!(hint.contains("copy the value"));
    }

    #[test]
    fn test_hint_double_mutable_borrow() {
        let error = "cannot borrow `s` as mutable more than once at a time";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Double Mutable Borrow"));
        assert!(hint.contains("ONE mutable reference"));
    }

    #[test]
    fn test_hint_return_local_reference() {
        let error = "cannot return reference to local variable `s`";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Returning Reference to Local"));
        assert!(hint.contains("owned type"));
    }

    #[test]
    fn test_hint_option_to_result() {
        let error = "expected `Result<i32, &str>`, found `Option<i32>`";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Option to Result"));
        assert!(hint.contains("ok_or"));
    }

    #[test]
    fn test_hint_question_mark_operator() {
        let error = "the `?` operator can only be used in a function that returns `Result`";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("? Operator Requires Result"));
        assert!(hint.contains("fn main()"));
    }

    #[test]
    fn test_hint_missing_clone() {
        let error = "the method `clone` exists but the following trait bounds were not satisfied: `Data` doesn't implement `Clone`";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Missing Clone"));
        assert!(hint.contains("#[derive(Clone)]"));
    }

    #[test]
    fn test_hint_missing_hash() {
        // Matches rustc output: "the trait `Hash` is not implemented for `Key`"
        let error = "the trait `Hash` is not implemented for `Key`";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Missing Hash"));
        assert!(hint.contains("Hash, PartialEq, Eq"));
    }

    #[test]
    fn test_hint_missing_ord() {
        // Matches rustc output: "the trait `Ord` is not implemented for `Score`"
        let error = "the trait `Ord` is not implemented for `Score`";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Missing Ord"));
        assert!(hint.contains("PartialEq, Eq, PartialOrd, Ord"));
    }

    #[test]
    fn test_hint_generic_trait_bound() {
        let error = "the trait bound `T: Display` is not satisfied";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.contains("Missing Trait Bound"));
    }

    #[test]
    fn test_hint_no_match() {
        let error = "some random error message";
        let hint = OllamaClient::get_error_hints(error);
        assert!(hint.is_empty());
    }
}
