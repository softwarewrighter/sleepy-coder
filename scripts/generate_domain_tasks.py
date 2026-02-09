#!/usr/bin/env python3
"""
Generate diverse Rust domain training tasks.

These are genuinely distinct skill domains that Share can learn:
1. Yew/WASM - Frontend components
2. REST Server - Actix/Axum patterns
3. DB Adapter - SQLx/Diesel queries
4. CLI - Command-line tools
5. Refactoring - Code improvement
6. Style/Metrics - Coding conventions
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sft" / "domains"


# Domain: Yew/WASM Frontend
YEW_WASM_TASKS = [
    {
        "id": "yew_001",
        "name": "Fix Yew component return type",
        "buggy": '''
use yew::prelude::*;

#[function_component(Counter)]
fn counter() {
    let count = use_state(|| 0);
    html! {
        <div>{"Count: "}{*count}</div>
    }
}
''',
        "fixed": '''
use yew::prelude::*;

#[function_component(Counter)]
fn counter() -> Html {
    let count = use_state(|| 0);
    html! {
        <div>{"Count: "}{*count}</div>
    }
}
''',
    },
    {
        "id": "yew_002",
        "name": "Fix Yew callback closure capture",
        "buggy": '''
use yew::prelude::*;

#[function_component(Button)]
fn button() -> Html {
    let count = use_state(|| 0);
    let onclick = Callback::from(|_| {
        count.set(*count + 1);
    });
    html! { <button {onclick}>{"Click"}</button> }
}
''',
        "fixed": '''
use yew::prelude::*;

#[function_component(Button)]
fn button() -> Html {
    let count = use_state(|| 0);
    let onclick = {
        let count = count.clone();
        Callback::from(move |_| {
            count.set(*count + 1);
        })
    };
    html! { <button {onclick}>{"Click"}</button> }
}
''',
    },
    {
        "id": "yew_003",
        "name": "Fix Yew props derive",
        "buggy": '''
use yew::prelude::*;

struct ButtonProps {
    label: String,
}

#[function_component(Button)]
fn button(props: &ButtonProps) -> Html {
    html! { <button>{&props.label}</button> }
}
''',
        "fixed": '''
use yew::prelude::*;

#[derive(Properties, PartialEq)]
struct ButtonProps {
    label: String,
}

#[function_component(Button)]
fn button(props: &ButtonProps) -> Html {
    html! { <button>{&props.label}</button> }
}
''',
    },
    {
        "id": "yew_004",
        "name": "Fix Yew use_effect dependency",
        "buggy": '''
use yew::prelude::*;

#[function_component(Logger)]
fn logger() -> Html {
    let count = use_state(|| 0);
    use_effect(|| {
        log::info!("Count changed: {}", *count);
    });
    html! { <div>{*count}</div> }
}
''',
        "fixed": '''
use yew::prelude::*;

#[function_component(Logger)]
fn logger() -> Html {
    let count = use_state(|| 0);
    {
        let count = count.clone();
        use_effect_with(count.clone(), move |count| {
            log::info!("Count changed: {}", **count);
            || ()
        });
    }
    html! { <div>{*count}</div> }
}
''',
    },
]


# Domain: REST Server (Axum)
AXUM_TASKS = [
    {
        "id": "axum_001",
        "name": "Fix Axum handler return type",
        "buggy": '''
use axum::{routing::get, Router};

async fn hello() {
    "Hello, World!"
}

fn app() -> Router {
    Router::new().route("/", get(hello))
}
''',
        "fixed": '''
use axum::{routing::get, Router};

async fn hello() -> &'static str {
    "Hello, World!"
}

fn app() -> Router {
    Router::new().route("/", get(hello))
}
''',
    },
    {
        "id": "axum_002",
        "name": "Fix Axum JSON extraction",
        "buggy": '''
use axum::{routing::post, Router, Json};
use serde::Deserialize;

struct CreateUser {
    name: String,
}

async fn create_user(user: Json<CreateUser>) -> String {
    format!("Created user: {}", user.name)
}
''',
        "fixed": '''
use axum::{routing::post, Router, Json};
use serde::Deserialize;

#[derive(Deserialize)]
struct CreateUser {
    name: String,
}

async fn create_user(Json(user): Json<CreateUser>) -> String {
    format!("Created user: {}", user.name)
}
''',
    },
    {
        "id": "axum_003",
        "name": "Fix Axum state extraction",
        "buggy": '''
use axum::{routing::get, Router, extract::State};
use std::sync::Arc;

struct AppState {
    db: String,
}

async fn handler(state: AppState) -> String {
    format!("DB: {}", state.db)
}

fn app() -> Router {
    let state = Arc::new(AppState { db: "postgres".into() });
    Router::new().route("/", get(handler)).with_state(state)
}
''',
        "fixed": '''
use axum::{routing::get, Router, extract::State};
use std::sync::Arc;

struct AppState {
    db: String,
}

async fn handler(State(state): State<Arc<AppState>>) -> String {
    format!("DB: {}", state.db)
}

fn app() -> Router {
    let state = Arc::new(AppState { db: "postgres".into() });
    Router::new().route("/", get(handler)).with_state(state)
}
''',
    },
    {
        "id": "axum_004",
        "name": "Fix Axum error handling",
        "buggy": '''
use axum::{routing::get, Router, http::StatusCode};

async fn might_fail() -> Result<String, String> {
    Err("Something went wrong".into())
}

async fn handler() -> String {
    might_fail().await?
}
''',
        "fixed": '''
use axum::{routing::get, Router, http::StatusCode, response::IntoResponse};

async fn might_fail() -> Result<String, String> {
    Err("Something went wrong".into())
}

async fn handler() -> Result<String, (StatusCode, String)> {
    might_fail().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))
}
''',
    },
]


# Domain: SQLx Database
SQLX_TASKS = [
    {
        "id": "sqlx_001",
        "name": "Fix SQLx query macro",
        "buggy": '''
use sqlx::PgPool;

async fn get_user(pool: &PgPool, id: i32) -> Result<String, sqlx::Error> {
    let row = sqlx::query("SELECT name FROM users WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await?;
    Ok(row.name)
}
''',
        "fixed": '''
use sqlx::PgPool;

async fn get_user(pool: &PgPool, id: i32) -> Result<String, sqlx::Error> {
    let row = sqlx::query!("SELECT name FROM users WHERE id = $1", id)
        .fetch_one(pool)
        .await?;
    Ok(row.name)
}
''',
    },
    {
        "id": "sqlx_002",
        "name": "Fix SQLx FromRow derive",
        "buggy": '''
use sqlx::PgPool;

struct User {
    id: i32,
    name: String,
}

async fn get_users(pool: &PgPool) -> Result<Vec<User>, sqlx::Error> {
    sqlx::query_as("SELECT id, name FROM users")
        .fetch_all(pool)
        .await
}
''',
        "fixed": '''
use sqlx::{PgPool, FromRow};

#[derive(FromRow)]
struct User {
    id: i32,
    name: String,
}

async fn get_users(pool: &PgPool) -> Result<Vec<User>, sqlx::Error> {
    sqlx::query_as("SELECT id, name FROM users")
        .fetch_all(pool)
        .await
}
''',
    },
    {
        "id": "sqlx_003",
        "name": "Fix SQLx transaction",
        "buggy": '''
use sqlx::PgPool;

async fn transfer(pool: &PgPool, from: i32, to: i32, amount: i32) -> Result<(), sqlx::Error> {
    sqlx::query!("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from)
        .execute(pool)
        .await?;
    sqlx::query!("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to)
        .execute(pool)
        .await?;
    Ok(())
}
''',
        "fixed": '''
use sqlx::PgPool;

async fn transfer(pool: &PgPool, from: i32, to: i32, amount: i32) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query!("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from)
        .execute(&mut *tx)
        .await?;
    sqlx::query!("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}
''',
    },
]


# Domain: CLI (clap)
CLI_TASKS = [
    {
        "id": "cli_001",
        "name": "Fix clap derive",
        "buggy": '''
use clap::Parser;

struct Args {
    #[arg(short, long)]
    name: String,
}

fn main() {
    let args = Args::parse();
    println!("Hello, {}!", args.name);
}
''',
        "fixed": '''
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    name: String,
}

fn main() {
    let args = Args::parse();
    println!("Hello, {}!", args.name);
}
''',
    },
    {
        "id": "cli_002",
        "name": "Fix clap subcommand",
        "buggy": '''
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

enum Commands {
    Add { name: String },
    Remove { id: i32 },
}
''',
        "fixed": '''
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add { name: String },
    Remove { id: i32 },
}
''',
    },
    {
        "id": "cli_003",
        "name": "Fix clap optional argument",
        "buggy": '''
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    config: Option<String>,
}

fn main() {
    let args = Args::parse();
    let config = args.config.unwrap();  // May panic
    println!("Config: {}", config);
}
''',
        "fixed": '''
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    config: Option<String>,
}

fn main() {
    let args = Args::parse();
    let config = args.config.unwrap_or_else(|| "default.toml".to_string());
    println!("Config: {}", config);
}
''',
    },
]


# Domain: Refactoring
REFACTORING_TASKS = [
    {
        "id": "refactor_001",
        "name": "Extract function from repeated code",
        "buggy": '''
fn process_users(users: Vec<User>) {
    for user in &users {
        println!("Processing: {}", user.name);
        if user.age > 18 {
            println!("Adult: {}", user.name);
        }
    }

    for user in &users {
        println!("Processing: {}", user.name);
        if user.age > 65 {
            println!("Senior: {}", user.name);
        }
    }
}
''',
        "fixed": '''
fn log_user(user: &User) {
    println!("Processing: {}", user.name);
}

fn process_users(users: Vec<User>) {
    for user in &users {
        log_user(user);
        if user.age > 18 {
            println!("Adult: {}", user.name);
        }
    }

    for user in &users {
        log_user(user);
        if user.age > 65 {
            println!("Senior: {}", user.name);
        }
    }
}
''',
    },
    {
        "id": "refactor_002",
        "name": "Use iterator methods instead of manual loop",
        "buggy": '''
fn sum_even(numbers: &[i32]) -> i32 {
    let mut sum = 0;
    for n in numbers {
        if n % 2 == 0 {
            sum += n;
        }
    }
    sum
}
''',
        "fixed": '''
fn sum_even(numbers: &[i32]) -> i32 {
    numbers.iter().filter(|n| *n % 2 == 0).sum()
}
''',
    },
    {
        "id": "refactor_003",
        "name": "Replace nested match with and_then",
        "buggy": '''
fn get_user_email(id: i32) -> Option<String> {
    match get_user(id) {
        Some(user) => {
            match user.email {
                Some(email) => Some(email),
                None => None,
            }
        }
        None => None,
    }
}
''',
        "fixed": '''
fn get_user_email(id: i32) -> Option<String> {
    get_user(id).and_then(|user| user.email)
}
''',
    },
]


# Domain: Style/Metrics (Coding Conventions from sw-checklist)
STYLE_TASKS = [
    {
        "id": "style_001",
        "name": "Split function exceeding 50 lines",
        "buggy": '''
fn process_order(order: Order) -> Result<(), Error> {
    // Validate order (too much in one function)
    if order.items.is_empty() {
        return Err(Error::EmptyOrder);
    }
    if order.total < 0.0 {
        return Err(Error::InvalidTotal);
    }

    // Calculate tax
    let tax = order.total * 0.08;
    let total_with_tax = order.total + tax;

    // Apply discount
    let discount = if total_with_tax > 100.0 { 0.1 } else { 0.0 };
    let final_total = total_with_tax * (1.0 - discount);

    // Save to database
    db::save_order(&order, final_total)?;

    // Send confirmation
    email::send_confirmation(&order.customer, final_total)?;

    Ok(())
}
''',
        "fixed": '''
fn validate_order(order: &Order) -> Result<(), Error> {
    if order.items.is_empty() {
        return Err(Error::EmptyOrder);
    }
    if order.total < 0.0 {
        return Err(Error::InvalidTotal);
    }
    Ok(())
}

fn calculate_total(order: &Order) -> f64 {
    let tax = order.total * 0.08;
    let total_with_tax = order.total + tax;
    let discount = if total_with_tax > 100.0 { 0.1 } else { 0.0 };
    total_with_tax * (1.0 - discount)
}

fn process_order(order: Order) -> Result<(), Error> {
    validate_order(&order)?;
    let final_total = calculate_total(&order);
    db::save_order(&order, final_total)?;
    email::send_confirmation(&order.customer, final_total)?;
    Ok(())
}
''',
    },
    {
        "id": "style_002",
        "name": "Reduce module function count (max 7 functions per module)",
        "buggy": '''
// lib.rs with 15+ functions - violates Module Function Count rule
pub fn create_user() { }
pub fn get_user() { }
pub fn update_user() { }
pub fn delete_user() { }
pub fn list_users() { }
pub fn create_post() { }
pub fn get_post() { }
pub fn update_post() { }
pub fn delete_post() { }
pub fn list_posts() { }
pub fn create_comment() { }
pub fn get_comment() { }
''',
        "fixed": '''
// lib.rs - organized into submodules
pub mod users;
pub mod posts;
pub mod comments;

// users.rs (max 7 functions)
pub fn create() { }
pub fn get() { }
pub fn update() { }
pub fn delete() { }
pub fn list() { }

// posts.rs (max 7 functions)
pub fn create() { }
pub fn get() { }
pub fn update() { }
pub fn delete() { }
pub fn list() { }

// comments.rs
pub fn create() { }
pub fn get() { }
''',
    },
    {
        "id": "style_003",
        "name": "Add AI agent instructions to CLI help",
        "buggy": '''
use clap::Parser;

#[derive(Parser)]
#[command(name = "mytool", version, about = "A CLI tool")]
struct Cli {
    #[arg(short, long)]
    verbose: bool,
}
''',
        "fixed": '''
use clap::Parser;

/// A CLI tool
///
/// # AI CODING AGENT INSTRUCTIONS
///
/// This tool processes files. When using as an AI agent:
/// - Use --verbose for detailed output
/// - Exit codes: 0 = success, 1 = error
/// - Output is JSON when --json flag is set
#[derive(Parser)]
#[command(name = "mytool", version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    verbose: bool,
}
''',
    },
    {
        "id": "style_004",
        "name": "Add complete version info",
        "buggy": '''
use clap::Parser;

#[derive(Parser)]
#[command(name = "mytool", version)]
struct Cli { }
''',
        "fixed": '''
use clap::Parser;

const VERSION_INFO: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\\nCopyright: 2024 Example Corp",
    "\\nLicense: MIT",
    "\\nRepository: https://github.com/example/mytool",
    "\\nBuild Host: ", env!("VERGEN_SYSINFO_NAME"),
    "\\nBuild Commit: ", env!("VERGEN_GIT_SHA"),
    "\\nBuild Time: ", env!("VERGEN_BUILD_TIMESTAMP"),
);

#[derive(Parser)]
#[command(name = "mytool", version = VERSION_INFO)]
struct Cli { }
''',
    },
    {
        "id": "style_005",
        "name": "Make --help longer than -h",
        "buggy": '''
#[derive(Parser)]
#[command(name = "mytool")]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}
''',
        "fixed": '''
#[derive(Parser)]
#[command(name = "mytool")]
struct Cli {
    /// Enable verbose output
    ///
    /// When enabled, the tool will print detailed progress information
    /// including file paths, processing times, and intermediate results.
    /// Useful for debugging and understanding tool behavior.
    #[arg(short, long, help = "Enable verbose output")]
    verbose: bool,
}
''',
    },
    {
        "id": "style_006",
        "name": "Split file exceeding 500 lines into modules",
        "buggy": '''
// main.rs - 700+ lines with multiple concerns
mod config { /* 100 lines */ }
mod database { /* 200 lines */ }
mod handlers { /* 300 lines */ }
mod utils { /* 100 lines */ }
fn main() { }
''',
        "fixed": '''
// main.rs - only entry point
mod config;
mod database;
mod handlers;
mod utils;

fn main() { }

// config.rs - separate file
// database.rs - separate file
// handlers.rs - separate file
// utils.rs - separate file
''',
    },
]


DOMAINS = {
    "yew_wasm": {
        "tasks": YEW_WASM_TASKS,
        "instruction": "You are a Yew/WASM expert. Fix the following Yew frontend component code.",
    },
    "axum_server": {
        "tasks": AXUM_TASKS,
        "instruction": "You are an Axum REST API expert. Fix the following server handler code.",
    },
    "sqlx_db": {
        "tasks": SQLX_TASKS,
        "instruction": "You are a SQLx database expert. Fix the following database code.",
    },
    "cli_clap": {
        "tasks": CLI_TASKS,
        "instruction": "You are a CLI development expert using clap. Fix the following CLI code.",
    },
    "refactoring": {
        "tasks": REFACTORING_TASKS,
        "instruction": "You are a Rust refactoring expert. Improve the following code structure.",
    },
    "style_metrics": {
        "tasks": STYLE_TASKS,
        "instruction": "You are a Rust code quality expert. Apply coding conventions to improve this code.",
    },
}


def generate_training_data():
    """Generate training data for all domains."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []

    for domain_name, domain in DOMAINS.items():
        domain_data = []
        instruction = domain["instruction"]

        for task in domain["tasks"]:
            example = {
                "instruction": instruction + "\nReturn ONLY the fixed Rust code without explanation.",
                "input": f"## Code:\n```rust\n{task['buggy'].strip()}\n```\n\n## Fixed Code:",
                "output": task["fixed"].strip(),
                "task_id": task["id"],
                "domain": domain_name,
            }
            domain_data.append(example)
            all_data.append(example)

        # Save domain-specific data
        with open(DATA_DIR / f"{domain_name}.jsonl", "w") as f:
            for ex in domain_data:
                f.write(json.dumps(ex) + "\n")

        print(f"Generated {len(domain_data)} examples for {domain_name}")

    # Save combined data
    with open(DATA_DIR / "all_domains.jsonl", "w") as f:
        for ex in all_data:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal: {len(all_data)} examples across {len(DOMAINS)} domains")
    print(f"Saved to: {DATA_DIR}")

    return all_data


if __name__ == "__main__":
    generate_training_data()
