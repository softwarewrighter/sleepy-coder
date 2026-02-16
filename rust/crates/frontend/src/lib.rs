//! Sleepy Coder Dashboard - Yew WASM Frontend
//!
//! This crate provides the web UI for the sleepy-coder dashboard.

mod app;
mod components;
mod pages;

pub use app::App;

use wasm_bindgen::prelude::*;

/// WASM entry point.
#[wasm_bindgen(start)]
pub fn main() {
    yew::Renderer::<App>::new().render();
}
