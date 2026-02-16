//! Statistics card component.

use yew::prelude::*;

/// Properties for StatCard component.
#[derive(Properties, PartialEq)]
pub struct StatCardProps {
    pub value: String,
    pub label: String,
}

/// Statistics card component.
#[function_component(StatCard)]
pub fn stat_card(props: &StatCardProps) -> Html {
    html! {
        <div class="card stat-card">
            <div class="stat-value">{ &props.value }</div>
            <div class="stat-label">{ &props.label }</div>
        </div>
    }
}
