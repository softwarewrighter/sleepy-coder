//! Tasks list page component.

use core_types::ErrorFamily;
use gloo_net::http::Request;
use web_types::TaskSummary;
use yew::prelude::*;

use crate::components::{Loading, TaskItem};

/// Tasks page component.
#[function_component(TasksPage)]
pub fn tasks_page() -> Html {
    let tasks = use_state(Vec::<TaskSummary>::new);
    let loading = use_state(|| true);
    let search = use_state(String::new);
    let family_filter = use_state(|| None::<ErrorFamily>);

    // Fetch tasks
    {
        let tasks = tasks.clone();
        let loading = loading.clone();
        let family = *family_filter;

        use_effect_with(family, move |_| {
            wasm_bindgen_futures::spawn_local(async move {
                let url = match family {
                    Some(f) => format!("/api/tasks?family={}", format!("{:?}", f).to_lowercase()),
                    None => "/api/tasks".to_string(),
                };

                match Request::get(&url).send().await {
                    Ok(resp) => {
                        if let Ok(data) = resp.json::<Vec<TaskSummary>>().await {
                            tasks.set(data);
                        }
                    }
                    Err(e) => {
                        gloo_timers::callback::Timeout::new(0, move || {
                            web_sys::console::error_1(
                                &format!("Failed to fetch tasks: {}", e).into(),
                            );
                        })
                        .forget();
                    }
                }
                loading.set(false);
            });
        });
    }

    let on_search_input = {
        let search = search.clone();
        Callback::from(move |e: InputEvent| {
            let input: web_sys::HtmlInputElement = e.target_unchecked_into();
            search.set(input.value());
        })
    };

    let on_family_change = {
        let family_filter = family_filter.clone();
        let loading = loading.clone();
        Callback::from(move |e: Event| {
            let select: web_sys::HtmlSelectElement = e.target_unchecked_into();
            let value = select.value();
            loading.set(true);
            family_filter.set(match value.as_str() {
                "borrow_checker" => Some(ErrorFamily::BorrowChecker),
                "lifetimes" => Some(ErrorFamily::Lifetimes),
                "trait_bounds" => Some(ErrorFamily::TraitBounds),
                "result_handling" => Some(ErrorFamily::ResultHandling),
                "type_mismatch" => Some(ErrorFamily::TypeMismatch),
                "other" => Some(ErrorFamily::Other),
                _ => None,
            });
        })
    };

    // Filter tasks by search
    let filtered_tasks: Vec<&TaskSummary> = tasks
        .iter()
        .filter(|t| {
            if search.is_empty() {
                true
            } else {
                let search_lower = search.to_lowercase();
                t.id.to_lowercase().contains(&search_lower)
                    || t.description.to_lowercase().contains(&search_lower)
            }
        })
        .collect();

    html! {
        <div>
            <h1>{"Tasks"}</h1>

            <div class="filter-bar">
                <select class="filter-select" onchange={on_family_change}>
                    <option value="">{"All Families"}</option>
                    <option value="borrow_checker">{"Borrow Checker"}</option>
                    <option value="lifetimes">{"Lifetimes"}</option>
                    <option value="trait_bounds">{"Trait Bounds"}</option>
                    <option value="result_handling">{"Result Handling"}</option>
                    <option value="type_mismatch">{"Type Mismatch"}</option>
                    <option value="other">{"Other"}</option>
                </select>
                <input
                    type="text"
                    class="search-input"
                    placeholder="Search tasks..."
                    oninput={on_search_input}
                />
            </div>

            if *loading {
                <Loading />
            } else if filtered_tasks.is_empty() {
                <div class="card">
                    <p>{"No tasks found."}</p>
                </div>
            } else {
                <div class="task-list">
                    { for filtered_tasks.iter().map(|task| {
                        html! { <TaskItem task={(*task).clone()} /> }
                    })}
                </div>
            }
        </div>
    }
}
