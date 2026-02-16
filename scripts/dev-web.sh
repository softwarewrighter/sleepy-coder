#!/bin/bash
# Development script for web UI
# Usage: ./scripts/dev-web.sh [server|frontend|both]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_ROOT/rust"

start_server() {
    echo "Starting backend server on http://localhost:3000..."
    cd "$RUST_DIR"
    cargo run -p server
}

start_frontend() {
    echo "Starting frontend dev server on http://localhost:8080..."
    cd "$RUST_DIR/crates/frontend"
    trunk serve
}

build_frontend() {
    echo "Building frontend for production..."
    cd "$RUST_DIR/crates/frontend"
    trunk build --release
}

case "${1:-both}" in
    server)
        start_server
        ;;
    frontend)
        start_frontend
        ;;
    build)
        build_frontend
        ;;
    both)
        echo "Starting both server and frontend..."
        echo "Run in separate terminals:"
        echo "  Terminal 1: ./scripts/dev-web.sh server"
        echo "  Terminal 2: ./scripts/dev-web.sh frontend"
        echo ""
        echo "Or use tmux/screen to run both in background."
        echo ""
        echo "Backend will be at: http://localhost:3000"
        echo "Frontend will be at: http://localhost:8080"
        ;;
    *)
        echo "Usage: $0 [server|frontend|build|both]"
        echo ""
        echo "Commands:"
        echo "  server   - Start the Axum backend server"
        echo "  frontend - Start the Trunk dev server with hot reload"
        echo "  build    - Build the frontend for production"
        echo "  both     - Show instructions for running both"
        exit 1
        ;;
esac
