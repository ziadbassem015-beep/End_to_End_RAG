#!/bin/bash
# run.sh - Central bootstrap script for the RAG Framework Infrastructure

echo "============================================================"
echo "      📖 RAG Decoupled Framework Control Script"
echo "============================================================"
echo "Choose an option to execute:"
echo "1) Run Backend API locally (FastAPI via uv)"
echo "2) Run React Frontend locally (Vite via npm)"
echo "3) Run both Backend and Frontend in Docker (via docker compose)"
echo "4) Run complete pytest verification suite"
echo "5) Run benchmark suite"
echo "6) Clean build caches and databases"
echo "============================================================"
read -p "Option [1-6]: " OPTION

case $OPTION in
    1)
        echo "Starting FastAPI Backend locally on port 8000..."
        export PYTHONPATH=.
        uv run uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    2)
        echo "Starting React Frontend locally on port 5173..."
        cd frontend
        npm run dev
        ;;
    3)
        echo "Building and starting Docker Containers..."
        docker compose up --build
        ;;
    4)
        echo "Running tests..."
        export PYTHONPATH=.
        uv run pytest
        ;;
    5)
        echo "Running benchmarks..."
        export PYTHONPATH=.
        uv run python rag_system/cli.py benchmark
        ;;
    6)
        echo "Cleaning caches, uploaded books, and vector stores..."
        rm -rf .pytest_cache .coverage data/uploads/* data/indices/faiss/* data/cache/* data/output/* data/reports/*
        echo "Cleanup complete."
        ;;
    *)
        echo "Invalid option."
        exit 1
        ;;
esac
