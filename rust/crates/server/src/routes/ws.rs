//! WebSocket handler for live fix streaming.

use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use web_types::{WsClientMessage, WsServerMessage};

use crate::state::AppState;

/// WebSocket upgrade handler.
pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle a WebSocket connection.
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => continue,
        };

        // Parse client message
        let client_msg: WsClientMessage = match serde_json::from_str(&msg) {
            Ok(m) => m,
            Err(e) => {
                let error = WsServerMessage::Error {
                    message: format!("Invalid message format: {}", e),
                };
                let _ = sender
                    .send(Message::Text(serde_json::to_string(&error).unwrap()))
                    .await;
                continue;
            }
        };

        // Handle the message
        match client_msg {
            WsClientMessage::StartFix { task_id } => {
                // Validate task exists
                if state.get_task(&task_id).is_none() {
                    let error = WsServerMessage::Error {
                        message: format!("Task not found: {}", task_id),
                    };
                    let _ = sender
                        .send(Message::Text(serde_json::to_string(&error).unwrap()))
                        .await;
                    continue;
                }

                // Create session
                let (session_id, mut session_receiver) =
                    state.create_session(task_id.clone()).await;

                // Send session started message
                let started = WsServerMessage::SessionStarted {
                    session_id,
                    task_id: task_id.clone(),
                };
                if sender
                    .send(Message::Text(serde_json::to_string(&started).unwrap()))
                    .await
                    .is_err()
                {
                    break;
                }

                // Get session sender for broadcasting
                let session_sender = state.get_session_sender(session_id).await;

                // Simulate fix progress (placeholder for actual agent integration)
                // In Phase 3, we'll hook this up to the actual AgentLoop
                if let Some(broadcast) = session_sender {
                    let _ = broadcast.send(WsServerMessage::AttemptStarted { attempt_idx: 0 });

                    // For now, just complete immediately
                    let _ = broadcast.send(WsServerMessage::SessionComplete {
                        session_id,
                        passed: false,
                        total_attempts: 0,
                        final_code: None,
                        diff_unified: None,
                    });
                }

                // Stream session events to client
                while let Ok(msg) = session_receiver.recv().await {
                    let json = serde_json::to_string(&msg).unwrap();
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }

                    // Stop streaming after session complete
                    if matches!(msg, WsServerMessage::SessionComplete { .. }) {
                        break;
                    }
                }

                state.remove_session(session_id).await;
            }
            WsClientMessage::CancelFix { session_id } => {
                state.remove_session(session_id).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_message_parsing() {
        let json = r#"{"type":"StartFix","payload":{"task_id":"task_001"}}"#;
        let msg: WsClientMessage = serde_json::from_str(json).unwrap();

        if let WsClientMessage::StartFix { task_id } = msg {
            assert_eq!(task_id, "task_001");
        } else {
            panic!("Wrong variant");
        }
    }
}
