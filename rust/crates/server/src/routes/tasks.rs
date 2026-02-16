//! Task-related API routes.

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use web_types::{ApiError, DiffView, TaskDetail, TaskFilter, TaskSummary};

use crate::state::AppState;

/// GET /api/tasks - List all tasks with optional filters.
pub async fn list_tasks(
    State(state): State<AppState>,
    Query(filter): Query<TaskFilter>,
) -> Json<Vec<TaskSummary>> {
    let tasks: Vec<TaskSummary> = state
        .tasks
        .iter()
        .filter(|task| {
            // Filter by family
            if let Some(family) = filter.family
                && task.family != family
            {
                return false;
            }

            // Filter by search text
            if let Some(ref search) = filter.search {
                let search_lower = search.to_lowercase();
                if !task.id.to_lowercase().contains(&search_lower)
                    && !task.description.to_lowercase().contains(&search_lower)
                {
                    return false;
                }
            }

            true
        })
        .map(TaskSummary::from)
        .collect();

    Json(tasks)
}

/// GET /api/tasks/:id - Get task detail.
pub async fn get_task(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<TaskDetail>, (StatusCode, Json<ApiError>)> {
    state
        .get_task(&id)
        .map(|task| Json(TaskDetail::from(task.clone())))
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiError::with_code(
                    format!("Task not found: {}", id),
                    "NOT_FOUND",
                )),
            )
        })
}

/// GET /api/tasks/:id/diff - Get diff view for a task.
pub async fn get_task_diff(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<DiffView>, (StatusCode, Json<ApiError>)> {
    let task = state.get_task(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError::with_code(
                format!("Task not found: {}", id),
                "NOT_FOUND",
            )),
        )
    })?;

    // Generate a simple unified diff
    let diff = generate_unified_diff(&task.buggy_code, &task.correct_code);

    Ok(Json(DiffView {
        task_id: id,
        original: task.buggy_code.clone(),
        fixed: task.correct_code.clone(),
        diff_unified: diff,
    }))
}

/// Generate a simple unified diff between two strings.
fn generate_unified_diff(original: &str, fixed: &str) -> String {
    let orig_lines: Vec<&str> = original.lines().collect();
    let fixed_lines: Vec<&str> = fixed.lines().collect();

    let mut diff = String::new();
    diff.push_str("--- original\n");
    diff.push_str("+++ fixed\n");

    // Simple line-by-line comparison
    let max_lines = orig_lines.len().max(fixed_lines.len());

    for i in 0..max_lines {
        let orig = orig_lines.get(i).copied();
        let fixed = fixed_lines.get(i).copied();

        match (orig, fixed) {
            (Some(o), Some(f)) if o == f => {
                diff.push_str(&format!(" {}\n", o));
            }
            (Some(o), Some(f)) => {
                diff.push_str(&format!("-{}\n", o));
                diff.push_str(&format!("+{}\n", f));
            }
            (Some(o), None) => {
                diff.push_str(&format!("-{}\n", o));
            }
            (None, Some(f)) => {
                diff.push_str(&format!("+{}\n", f));
            }
            (None, None) => {}
        }
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_unified_diff() {
        let original = "fn main() {\n    let x = 1;\n}";
        let fixed = "fn main() {\n    let x = 2;\n}";

        let diff = generate_unified_diff(original, fixed);

        assert!(diff.contains("--- original"));
        assert!(diff.contains("+++ fixed"));
        assert!(diff.contains("-    let x = 1;"));
        assert!(diff.contains("+    let x = 2;"));
    }

    #[test]
    fn test_generate_unified_diff_identical() {
        let code = "fn main() {}";
        let diff = generate_unified_diff(code, code);

        assert!(!diff.contains("-fn main"));
        assert!(!diff.contains("+fn main"));
    }
}
