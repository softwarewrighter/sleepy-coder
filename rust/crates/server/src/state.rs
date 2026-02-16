//! Application state for the web server.

use core_types::Task;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;
use web_types::WsServerMessage;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    /// All available tasks.
    pub tasks: Arc<Vec<Task>>,
    /// Task lookup by ID.
    pub tasks_by_id: Arc<HashMap<String, Task>>,
    /// Active fix sessions.
    pub sessions: Arc<RwLock<HashMap<Uuid, FixSession>>>,
}

/// An active fix session.
pub struct FixSession {
    /// Task ID for this session (used for future agent integration).
    #[allow(dead_code)]
    pub task_id: String,
    pub sender: broadcast::Sender<WsServerMessage>,
}

impl AppState {
    /// Create a new app state with the given tasks.
    pub fn new(tasks: Vec<Task>) -> Self {
        let tasks_by_id: HashMap<String, Task> =
            tasks.iter().map(|t| (t.id.clone(), t.clone())).collect();

        Self {
            tasks: Arc::new(tasks),
            tasks_by_id: Arc::new(tasks_by_id),
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: &str) -> Option<&Task> {
        self.tasks_by_id.get(id)
    }

    /// Create a new fix session.
    pub async fn create_session(
        &self,
        task_id: String,
    ) -> (Uuid, broadcast::Receiver<WsServerMessage>) {
        let session_id = Uuid::new_v4();
        let (sender, receiver) = broadcast::channel(32);

        let session = FixSession { task_id, sender };

        self.sessions.write().await.insert(session_id, session);

        (session_id, receiver)
    }

    /// Get a session's sender for broadcasting messages.
    pub async fn get_session_sender(
        &self,
        session_id: Uuid,
    ) -> Option<broadcast::Sender<WsServerMessage>> {
        self.sessions
            .read()
            .await
            .get(&session_id)
            .map(|s| s.sender.clone())
    }

    /// Remove a session.
    pub async fn remove_session(&self, session_id: Uuid) {
        self.sessions.write().await.remove(&session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_types::ErrorFamily;

    #[test]
    fn test_app_state_creation() {
        let tasks = vec![Task::new(
            "test_001".to_string(),
            ErrorFamily::BorrowChecker,
            "Test".to_string(),
            "buggy".to_string(),
            "fixed".to_string(),
        )];

        let state = AppState::new(tasks);

        assert_eq!(state.tasks.len(), 1);
        assert!(state.get_task("test_001").is_some());
        assert!(state.get_task("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_session_creation() {
        let state = AppState::new(vec![]);

        let (session_id, _receiver) = state.create_session("task_001".to_string()).await;

        assert!(state.get_session_sender(session_id).await.is_some());

        state.remove_session(session_id).await;

        assert!(state.get_session_sender(session_id).await.is_none());
    }
}
