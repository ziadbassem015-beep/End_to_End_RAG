@echo off
rem run.bat - Central bootstrap script for Windows Host

echo ============================================================
echo       📖 RAG Decoupled Framework Control Script
echo ============================================================
echo Choose an option to execute:
echo 1) Run Backend API locally (FastAPI via uv)
echo 2) Run React Frontend locally (Vite via npm)
echo 3) Run both Backend and Frontend in Docker (via docker compose)
echo 4) Run complete pytest verification suite
echo 5) Run benchmark suite
echo 6) Clean build caches and databases
echo ============================================================
set /p OPTION="Option [1-6]: "

if "%OPTION%"=="1" (
    echo Starting FastAPI Backend locally on port 8000...
    set PYTHONPATH=.
    uv run uvicorn rag_system.api.main:app --host 0.0.0.0 --port 8000 --reload
    goto end
)
if "%OPTION%"=="2" (
    echo Starting React Frontend locally on port 5173...
    cd frontend
    npm run dev
    goto end
)
if "%OPTION%"=="3" (
    echo Building and starting Docker Containers...
    docker compose up --build
    goto end
)
if "%OPTION%"=="4" (
    echo Running tests...
    set PYTHONPATH=.
    uv run pytest
    goto end
)
if "%OPTION%"=="5" (
    echo Running benchmarks...
    set PYTHONPATH=.
    uv run python rag_system/cli.py benchmark
    goto end
)
if "%OPTION%"=="6" (
    echo Cleaning caches, uploaded books, and vector stores...
    rmdir /s /q .pytest_cache
    del /q .coverage
    del /q data\uploads\*
    rmdir /s /q data\indices\faiss
    mkdir data\indices\faiss
    rmdir /s /q data\cache
    mkdir data\cache
    rmdir /s /q data\output
    mkdir data\output
    rmdir /s /q data\reports
    mkdir data\reports
    echo Cleanup complete.
    goto end
)

echo Invalid option.
exit /b 1

:end
pause
