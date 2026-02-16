//! Diff view component for displaying code changes.

use yew::prelude::*;

/// Properties for DiffView component.
#[derive(Properties, PartialEq)]
pub struct DiffViewProps {
    pub diff: String,
}

/// Diff view component.
#[function_component(DiffView)]
pub fn diff_view(props: &DiffViewProps) -> Html {
    let lines: Vec<Html> = props
        .diff
        .lines()
        .map(|line| {
            let class = if line.starts_with('+') && !line.starts_with("+++") {
                "diff-line added"
            } else if line.starts_with('-') && !line.starts_with("---") {
                "diff-line removed"
            } else {
                "diff-line context"
            };

            html! {
                <div class={class}>{ line }</div>
            }
        })
        .collect();

    html! {
        <div class="diff-view code-block">
            { for lines }
        </div>
    }
}
