//! Loading spinner component.

use yew::prelude::*;

/// Loading spinner component.
#[function_component(Loading)]
pub fn loading() -> Html {
    html! {
        <div class="loading">
            <div class="spinner"></div>
        </div>
    }
}
