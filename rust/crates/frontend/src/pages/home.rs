//! Home page component.

use gloo_net::http::Request;
use web_types::AnalyticsSummary;
use yew::prelude::*;
use yew_router::prelude::*;

use crate::app::Route;
use crate::components::{Loading, StatCard};

/// Home page component.
#[function_component(HomePage)]
pub fn home_page() -> Html {
    let analytics = use_state(|| None::<AnalyticsSummary>);
    let loading = use_state(|| true);

    {
        let analytics = analytics.clone();
        let loading = loading.clone();

        use_effect_with((), move |_| {
            wasm_bindgen_futures::spawn_local(async move {
                match Request::get("/api/analytics").send().await {
                    Ok(resp) => {
                        if let Ok(data) = resp.json::<AnalyticsSummary>().await {
                            analytics.set(Some(data));
                        }
                    }
                    Err(e) => {
                        gloo_timers::callback::Timeout::new(0, move || {
                            web_sys::console::error_1(
                                &format!("Failed to fetch analytics: {}", e).into(),
                            );
                        })
                        .forget();
                    }
                }
                loading.set(false);
            });
        });
    }

    if *loading {
        return html! { <Loading /> };
    }

    let stats = analytics.as_ref();

    html! {
        <div>
            <h1>{"Sleepy Coder Dashboard"}</h1>
            <p class="text-secondary" style="margin-bottom: 2rem;">
                {"Continual learning agent for fixing Rust compilation errors"}
            </p>

            <div class="stats-grid">
                <StatCard
                    value={stats.map(|s| s.total_tasks.to_string()).unwrap_or_else(|| "-".to_string())}
                    label={"Total Tasks"}
                />
                <StatCard
                    value={stats.map(|s| s.passed_tasks.to_string()).unwrap_or_else(|| "-".to_string())}
                    label={"Passed"}
                />
                <StatCard
                    value={stats.map(|s| format!("{:.1}%", s.pass_rate * 100.0)).unwrap_or_else(|| "-".to_string())}
                    label={"Pass Rate"}
                />
                <StatCard
                    value={stats.map(|s| format!("{:.1}", s.avg_steps_to_green)).unwrap_or_else(|| "-".to_string())}
                    label={"Avg Steps"}
                />
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">{"Quick Actions"}</h2>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <Link<Route> to={Route::Tasks} classes="btn btn-primary">
                        {"Browse Tasks"}
                    </Link<Route>>
                    <Link<Route> to={Route::Dashboard} classes="btn btn-secondary">
                        {"View Analytics"}
                    </Link<Route>>
                </div>
            </div>

            if let Some(analytics) = stats {
                <div class="card">
                    <div class="card-header">
                        <h2 class="card-title">{"Error Family Breakdown"}</h2>
                    </div>
                    <div class="task-list">
                        { for analytics.error_family_breakdown.iter().map(|family| {
                            html! {
                                <div class="task-item" style="cursor: default;">
                                    <div class="task-info">
                                        <div class="task-id">{ format!("{:?}", family.family) }</div>
                                        <div class="task-description">
                                            { format!("{} tasks", family.count) }
                                        </div>
                                    </div>
                                    <div class="task-family">
                                        { format!("{:.0}%", family.pass_rate * 100.0) }
                                    </div>
                                </div>
                            }
                        })}
                    </div>
                </div>
            }
        </div>
    }
}
