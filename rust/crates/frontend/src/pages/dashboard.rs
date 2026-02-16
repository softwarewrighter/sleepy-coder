//! Dashboard page component with analytics.

use gloo_net::http::Request;
use web_types::AnalyticsSummary;
use yew::prelude::*;

use crate::components::{Loading, StatCard};

/// Dashboard page component.
#[function_component(DashboardPage)]
pub fn dashboard_page() -> Html {
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

    let Some(stats) = analytics.as_ref() else {
        return html! {
            <div class="card">
                <p>{"Failed to load analytics."}</p>
            </div>
        };
    };

    html! {
        <div>
            <h1>{"Analytics Dashboard"}</h1>

            <div class="stats-grid">
                <StatCard
                    value={stats.total_tasks.to_string()}
                    label={"Total Tasks"}
                />
                <StatCard
                    value={stats.passed_tasks.to_string()}
                    label={"Passed Tasks"}
                />
                <StatCard
                    value={stats.failed_tasks.to_string()}
                    label={"Failed Tasks"}
                />
                <StatCard
                    value={format!("{:.1}%", stats.pass_rate * 100.0)}
                    label={"Pass Rate"}
                />
            </div>

            <div class="stats-grid">
                <StatCard
                    value={stats.total_attempts.to_string()}
                    label={"Total Attempts"}
                />
                <StatCard
                    value={format!("{:.1}", stats.avg_steps_to_green)}
                    label={"Avg Steps to Green"}
                />
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">{"Error Family Breakdown"}</h2>
                </div>

                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    { for stats.error_family_breakdown.iter().map(|family| {
                        let percentage = if stats.total_tasks > 0 {
                            (family.count as f64 / stats.total_tasks as f64) * 100.0
                        } else {
                            0.0
                        };

                        html! {
                            <div>
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                    <span>{ format!("{:?}", family.family) }</span>
                                    <span style="color: var(--text-secondary);">
                                        { format!("{} ({:.0}%)", family.count, percentage) }
                                    </span>
                                </div>
                                <div class="progress-bar">
                                    <div
                                        class="progress-bar-fill"
                                        style={format!("width: {}%", percentage)}
                                    />
                                </div>
                                <div style="font-size: 0.75rem; color: var(--text-secondary);">
                                    { format!("{}/{} passed ({:.0}%)", family.passed, family.count, family.pass_rate * 100.0) }
                                </div>
                            </div>
                        }
                    })}
                </div>
            </div>
        </div>
    }
}
