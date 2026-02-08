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
        let prompt = format!(
            r#"Fix the following Rust code that has a compilation error.

## Buggy Code:
```rust
{buggy_code}
```

## Compiler Error:
{error_message}

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
}
