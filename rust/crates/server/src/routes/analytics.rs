//! Analytics API routes.

use axum::{Json, extract::State};
use core_types::ErrorFamily;
use web_types::{AnalyticsSummary, FamilyStats};

use crate::state::AppState;

/// GET /api/analytics - Get analytics summary.
pub async fn get_analytics(State(state): State<AppState>) -> Json<AnalyticsSummary> {
    let tasks = &state.tasks;

    // Count tasks by family
    let families = [
        ErrorFamily::BorrowChecker,
        ErrorFamily::Lifetimes,
        ErrorFamily::TraitBounds,
        ErrorFamily::ResultHandling,
        ErrorFamily::TypeMismatch,
        ErrorFamily::Other,
    ];

    let family_breakdown: Vec<FamilyStats> = families
        .iter()
        .map(|&family| {
            let count = tasks.iter().filter(|t| t.family == family).count() as u32;
            FamilyStats {
                family,
                count,
                passed: 0, // Will be populated when we have episode data
                pass_rate: 0.0,
            }
        })
        .filter(|s| s.count > 0)
        .collect();

    let total_tasks = tasks.len() as u32;

    Json(AnalyticsSummary {
        total_tasks,
        passed_tasks: 0,   // Will be populated from episodes
        failed_tasks: 0,   // Will be populated from episodes
        pass_rate: 0.0,    // Will be calculated
        total_attempts: 0, // Will be populated from episodes
        avg_steps_to_green: 0.0,
        error_family_breakdown: family_breakdown,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use core_types::Task;

    #[tokio::test]
    async fn test_get_analytics() {
        let tasks = vec![
            Task::new(
                "t1".to_string(),
                ErrorFamily::BorrowChecker,
                "Test 1".to_string(),
                "buggy".to_string(),
                "fixed".to_string(),
            ),
            Task::new(
                "t2".to_string(),
                ErrorFamily::BorrowChecker,
                "Test 2".to_string(),
                "buggy".to_string(),
                "fixed".to_string(),
            ),
            Task::new(
                "t3".to_string(),
                ErrorFamily::Lifetimes,
                "Test 3".to_string(),
                "buggy".to_string(),
                "fixed".to_string(),
            ),
        ];

        let state = AppState::new(tasks);
        let Json(analytics) = get_analytics(State(state)).await;

        assert_eq!(analytics.total_tasks, 3);
        assert_eq!(analytics.error_family_breakdown.len(), 2);

        let borrow_stats = analytics
            .error_family_breakdown
            .iter()
            .find(|s| s.family == ErrorFamily::BorrowChecker)
            .unwrap();
        assert_eq!(borrow_stats.count, 2);
    }
}
