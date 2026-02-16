//! Task detail page component.

use gloo_net::http::Request;
use web_types::{DiffView as DiffViewData, TaskDetail};
use yew::prelude::*;

use crate::components::{CodeBlock, DiffView, Loading};

/// Properties for TaskDetailPage.
#[derive(Properties, PartialEq)]
pub struct TaskDetailPageProps {
    pub task_id: String,
}

/// Task detail page component.
#[function_component(TaskDetailPage)]
pub fn task_detail_page(props: &TaskDetailPageProps) -> Html {
    let task = use_state(|| None::<TaskDetail>);
    let diff = use_state(|| None::<DiffViewData>);
    let loading = use_state(|| true);
    let view_mode = use_state(|| ViewMode::SideBySide);
    let task_id = props.task_id.clone();

    // Fetch task and diff
    {
        let task = task.clone();
        let diff = diff.clone();
        let loading = loading.clone();
        let task_id = task_id.clone();

        use_effect_with(task_id.clone(), move |_| {
            wasm_bindgen_futures::spawn_local(async move {
                // Fetch task detail
                if let Ok(resp) = Request::get(&format!("/api/tasks/{}", task_id))
                    .send()
                    .await
                    && let Ok(data) = resp.json::<TaskDetail>().await
                {
                    task.set(Some(data));
                }

                // Fetch diff
                if let Ok(resp) = Request::get(&format!("/api/tasks/{}/diff", task_id))
                    .send()
                    .await
                    && let Ok(data) = resp.json::<DiffViewData>().await
                {
                    diff.set(Some(data));
                }

                loading.set(false);
            });
        });
    }

    let on_view_mode_change = {
        let view_mode = view_mode.clone();
        Callback::from(move |mode: ViewMode| {
            view_mode.set(mode);
        })
    };

    if *loading {
        return html! { <Loading /> };
    }

    let Some(task_data) = task.as_ref() else {
        return html! {
            <div class="card">
                <h1>{"Task Not Found"}</h1>
                <p>{"The requested task could not be found."}</p>
            </div>
        };
    };

    let family_str = format!("{:?}", task_data.family);

    html! {
        <div>
            <div class="card">
                <div class="card-header">
                    <h1 class="card-title">{ &task_data.id }</h1>
                    <span class="task-family">{ &family_str }</span>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    { &task_data.description }
                </p>
                <button class="btn btn-primary">
                    {"Run Fix"}
                </button>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">{"Code View"}</h2>
                    <div>
                        <button
                            class={if *view_mode == ViewMode::SideBySide { "btn btn-primary" } else { "btn btn-secondary" }}
                            onclick={on_view_mode_change.reform(|_| ViewMode::SideBySide)}
                            style="margin-right: 0.5rem;"
                        >
                            {"Side by Side"}
                        </button>
                        <button
                            class={if *view_mode == ViewMode::Diff { "btn btn-primary" } else { "btn btn-secondary" }}
                            onclick={on_view_mode_change.reform(|_| ViewMode::Diff)}
                        >
                            {"Unified Diff"}
                        </button>
                    </div>
                </div>

                if *view_mode == ViewMode::SideBySide {
                    <div class="live-fix-container">
                        <div>
                            <h3 style="margin-bottom: 0.5rem; color: var(--accent-error);">{"Buggy Code"}</h3>
                            <CodeBlock code={task_data.buggy_code.clone()} />
                        </div>
                        <div>
                            <h3 style="margin-bottom: 0.5rem; color: var(--accent-success);">{"Correct Code"}</h3>
                            <CodeBlock code={task_data.correct_code.clone()} />
                        </div>
                    </div>
                } else if let Some(diff_data) = diff.as_ref() {
                    <DiffView diff={diff_data.diff_unified.clone()} />
                }
            </div>
        </div>
    }
}

/// View mode for code display.
#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    SideBySide,
    Diff,
}
