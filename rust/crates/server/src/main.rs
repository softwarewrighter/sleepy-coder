//! Sleepy-coder web server.
//!
//! Provides a REST API and WebSocket interface for the sleepy-coder dashboard.

mod routes;
mod state;

use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

use routes::{get_analytics, get_task, get_task_diff, list_tasks, ws_handler};
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load tasks from the koans
    let tasks = tasks_rust_koans::load_builtin_koans();
    println!("Loaded {} tasks", tasks.len());

    // Create app state
    let state = AppState::new(tasks);

    // Build CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build API routes
    let api_routes = Router::new()
        .route("/tasks", get(list_tasks))
        .route("/tasks/:id", get(get_task))
        .route("/tasks/:id/diff", get(get_task_diff))
        .route("/analytics", get(get_analytics))
        .route("/fix/start", post(start_fix_placeholder));

    // Build main router
    let app = Router::new()
        .nest("/api", api_routes)
        .route("/ws", get(ws_handler))
        // Serve static files from frontend dist (when built)
        .fallback_service(ServeDir::new("../frontend/dist").append_index_html_on_directories(true))
        .layer(cors)
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 5970));
    println!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Placeholder for POST /api/fix/start (returns session ID for polling).
async fn start_fix_placeholder(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::Json(req): axum::Json<web_types::StartFixRequest>,
) -> Result<
    axum::Json<web_types::StartFixResponse>,
    (axum::http::StatusCode, axum::Json<web_types::ApiError>),
> {
    // Validate task exists
    if state.get_task(&req.task_id).is_none() {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(web_types::ApiError::with_code(
                format!("Task not found: {}", req.task_id),
                "NOT_FOUND",
            )),
        ));
    }

    // Create session (for WebSocket to connect to)
    let (session_id, _) = state.create_session(req.task_id).await;

    Ok(axum::Json(web_types::StartFixResponse { session_id }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_starts() {
        let tasks = tasks_rust_koans::load_builtin_koans();
        let _state = AppState::new(tasks);
        // Basic smoke test
    }
}
