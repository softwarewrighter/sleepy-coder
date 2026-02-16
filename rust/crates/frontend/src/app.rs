//! Main application component with routing.

use yew::prelude::*;
use yew_router::prelude::*;

use crate::pages::{DashboardPage, HomePage, TaskDetailPage, TasksPage};

/// Application routes.
#[derive(Clone, Routable, PartialEq)]
pub enum Route {
    #[at("/")]
    Home,
    #[at("/tasks")]
    Tasks,
    #[at("/tasks/:id")]
    TaskDetail { id: String },
    #[at("/dashboard")]
    Dashboard,
    #[not_found]
    #[at("/404")]
    NotFound,
}

/// Route switch function.
fn switch(routes: Route) -> Html {
    match routes {
        Route::Home => html! { <HomePage /> },
        Route::Tasks => html! { <TasksPage /> },
        Route::TaskDetail { id } => html! { <TaskDetailPage task_id={id} /> },
        Route::Dashboard => html! { <DashboardPage /> },
        Route::NotFound => html! {
            <div class="card">
                <h1>{"404 - Page Not Found"}</h1>
                <p>{"The page you're looking for doesn't exist."}</p>
            </div>
        },
    }
}

/// Main application component.
#[function_component(App)]
pub fn app() -> Html {
    html! {
        <BrowserRouter>
            <div class="app-container">
                <Sidebar />
                <main class="main-content">
                    <Switch<Route> render={switch} />
                </main>
            </div>
        </BrowserRouter>
    }
}

/// Sidebar navigation component.
#[function_component(Sidebar)]
fn sidebar() -> Html {
    html! {
        <aside class="sidebar">
            <Link<Route> to={Route::Home} classes="nav-brand">
                {"Sleepy Coder"}
            </Link<Route>>
            <nav>
                <ul class="nav-links">
                    <li>
                        <Link<Route> to={Route::Home}>
                            {"Home"}
                        </Link<Route>>
                    </li>
                    <li>
                        <Link<Route> to={Route::Tasks}>
                            {"Tasks"}
                        </Link<Route>>
                    </li>
                    <li>
                        <Link<Route> to={Route::Dashboard}>
                            {"Dashboard"}
                        </Link<Route>>
                    </li>
                </ul>
            </nav>
        </aside>
    }
}
