//! Task list item component.

use web_types::TaskSummary;
use yew::prelude::*;
use yew_router::prelude::*;

use crate::app::Route;

/// Properties for TaskItem component.
#[derive(Properties, PartialEq)]
pub struct TaskItemProps {
    pub task: TaskSummary,
}

/// Task list item component.
#[function_component(TaskItem)]
pub fn task_item(props: &TaskItemProps) -> Html {
    let task = &props.task;

    let status_class = match task.passed {
        Some(true) => "task-status passed",
        Some(false) => "task-status failed",
        None => "task-status pending",
    };

    let family_str = format!("{:?}", task.family).to_lowercase();

    html! {
        <Link<Route> to={Route::TaskDetail { id: task.id.clone() }}>
            <div class="task-item">
                <div class={status_class}></div>
                <div class="task-info">
                    <div class="task-id">{ &task.id }</div>
                    <div class="task-description">{ &task.description }</div>
                </div>
                <div class="task-family">{ family_str }</div>
            </div>
        </Link<Route>>
    }
}
