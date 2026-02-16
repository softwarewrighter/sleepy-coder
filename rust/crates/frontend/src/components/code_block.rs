//! Code block component for displaying source code.

use yew::prelude::*;

/// Properties for CodeBlock component.
#[derive(Properties, PartialEq)]
pub struct CodeBlockProps {
    pub code: String,
    #[prop_or_default]
    pub language: Option<String>,
}

/// Code block component.
#[function_component(CodeBlock)]
pub fn code_block(props: &CodeBlockProps) -> Html {
    html! {
        <div class="code-block">
            <pre>
                <code>{ &props.code }</code>
            </pre>
        </div>
    }
}
