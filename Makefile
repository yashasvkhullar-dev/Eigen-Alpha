# ══════════════════════════════════════════════════════════════════════
# EigenAlpha — Makefile
# ══════════════════════════════════════════════════════════════════════
# Usage:
#   make install     — create venv and install dependencies
#   make run         — run the full pipeline
#   make test        — run all unit tests
#   make lint        — run black + flake8
#   make clean       — remove caches and generated outputs
#   make dashboard   — launch API + frontend
#   make help        — show this help
# ══════════════════════════════════════════════════════════════════════

PYTHON   ?= python
PIP      ?= pip
VENV_DIR ?= .venv
PYTEST   ?= pytest

.PHONY: help install run test lint format clean dashboard

help:  ## Show this help
	@echo.
	@echo   EigenAlpha — Available commands:
	@echo   ─────────────────────────────────────────────
	@echo   make install     Create venv + install deps
	@echo   make run         Run the full pipeline
	@echo   make test        Run pytest test suite
	@echo   make lint        Run flake8 linting
	@echo   make format      Run black code formatter
	@echo   make clean       Remove caches and outputs
	@echo   make dashboard   Launch API + frontend
	@echo   ─────────────────────────────────────────────

install:  ## Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/Scripts/pip install --upgrade pip
	$(VENV_DIR)/Scripts/pip install -r requirements.txt
	$(VENV_DIR)/Scripts/pip install -e ".[dev]"
	@echo [OK] Virtual environment ready. Activate with: $(VENV_DIR)\Scripts\activate

run:  ## Run the full EigenAlpha pipeline
	$(PYTHON) pipeline.py

test:  ## Run all unit tests
	$(PYTEST) tests/ -v --tb=short

lint:  ## Run flake8 linting
	flake8 --max-line-length=100 --exclude=.venv,__pycache__,frontend,node_modules .

format:  ## Format code with black
	black --line-length=100 --target-version=py310 .

clean:  ## Remove generated files, caches, and temporary data
	@echo Cleaning __pycache__ directories...
	@for /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@echo Cleaning pytest cache...
	@if exist .pytest_cache rd /s /q .pytest_cache
	@echo Cleaning outputs...
	@if exist outputs\tearsheet.png del outputs\tearsheet.png
	@if exist outputs\plots rd /s /q outputs\plots
	@if exist outputs\*.parquet del outputs\*.parquet
	@echo Cleaning data cache...
	@if exist data_cache rd /s /q data_cache
	@echo [OK] Clean complete.

dashboard:  ## Launch the FastAPI backend and Next.js frontend
	@echo Starting EigenAlpha dashboard...
	@echo Backend: http://localhost:8000
	@echo Frontend: http://localhost:3000
	$(PYTHON) -c "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8000)" &
	cd frontend && npm run dev
