# Makefile for Federated Learning with Fog Computing Demo
#
# This Makefile provides convenient commands for development, testing,
# and deployment following modern Python development practices.

PYTHON = python
VENV_PYTHON = .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = .venv/Scripts/python.exe
else
    VENV_PYTHON = .venv/bin/python
endif

.PHONY: help install install-dev test test-cov test-unit test-integration lint format type-check clean build docs serve-docs pre-commit install-pre-commit quality fix dev-setup dev-check

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install package and dependencies
	$(PYTHON) -m pip install -e .

install-dev: ## Install package with development dependencies
	$(PYTHON) -m pip install -e ".[dev,test]"

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

# Testing
test: ## Run all tests
	$(PYTHON) -m pytest

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest -m "not integration"

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest -m integration

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term-missing

test-verbose: ## Run tests with verbose output
	$(PYTHON) -m pytest -v

# Code Quality
lint: ## Run all linters
	ruff check .
	flake8 .

format: ## Format code with black and isort
	black .
	isort .

type-check: ## Run mypy type checking
	mypy .

quality: lint type-check ## Run all code quality checks

fix: ## Auto-fix code issues
	ruff check . --fix
	black .
	isort .

# Pre-commit
pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Cleaning
clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf */*/__pycache__/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-all: clean ## Clean everything including virtual environment
	rm -rf .venv/
	rm -rf venv/

# Build
build: clean ## Build package distribution
	$(PYTHON) -m build

# Development workflow
dev-setup: install-dev install-pre-commit ## Set up development environment

dev-check: format quality test ## Run full development check

# Demo commands
run-demo: ## Run complete federated learning demo
	$(PYTHON) -c "from compare_models import ModelComparator; comp = ModelComparator(); comp.run_robust_comparison(n_cv_folds=2)"

run-quick: ## Run quick comparison demo
	$(PYTHON) quick_comparison.py

# Documentation
docs: ## Generate documentation (if sphinx is configured)
	sphinx-build -b html docs/ docs/_build/html

serve-docs: docs ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# CI/CD simulation
ci: dev-check build ## Simulate CI pipeline
	@echo "✅ CI pipeline completed successfully!"

# Utility
deps-update: ## Update all dependencies
	pip install --upgrade pip
	pip install --upgrade -e ".[dev,test]"

deps-list: ## List installed packages
	pip list

deps-outdated: ## Show outdated packages
	pip list --outdated

# Git helpers
git-clean: ## Remove untracked files
	git clean -fd

git-status: ## Show git status with useful info
	@echo "=== Git Status ==="
	git status --short
	@echo
	@echo "=== Recent Commits ==="
	git log --oneline -5

# Environment info
info: ## Show environment information
	@echo "=== Python Environment ==="
	$(PYTHON) --version
	which $(PYTHON)
	@echo
	@echo "=== Package Info ==="
	pip show flower-basic 2>/dev/null || echo "Package not installed"
	@echo
	@echo "=== System Info ==="
	uname -a 2>/dev/null || echo "Not on Unix-like system"

test-fast:  ## Ejecutar tests rápidos solamente
	$(PYTHON) -m pytest tests/ -v -m "not slow"

quality:  ## Ejecutar todas las verificaciones de calidad
	make format
	make lint
	make type-check
	make test-unit

clean:  ## Limpiar archivos temporales
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "✅ Archivos temporales limpiados"

demo:  ## Ejecutar demo del sistema (requiere componentes manuales)
	@echo "🚀 Para ejecutar la demo completa:"
	@echo "1. Terminal 1: mosquitto -v"
	@echo "2. Terminal 2: $(PYTHON) server.py"
	@echo "3. Terminal 3: $(PYTHON) broker_fog.py"
	@echo "4. Terminal 4: $(PYTHON) fog_flower_client.py"
	@echo "5. Terminal 5+: $(PYTHON) client.py (múltiples instancias)"

system-test:  ## Verificar que el sistema puede importarse
	$(PYTHON) test_system.py

docs:  ## Generar documentación (placeholder)
	@echo "📚 Documentación disponible en README.md"
	@echo "🔍 Para documentación detallada de API, considera agregar Sphinx"

dev:  ## Configurar entorno de desarrollo completo
	make setup
	make install
	make quality
	@echo "✅ Entorno de desarrollo listo!"

# Comandos de CI/CD
ci-test:  ## Tests para CI/CD
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not integration"

ci-quality:  ## Verificaciones de calidad para CI/CD
	$(PYTHON) -m isort . --check-only
	$(PYTHON) -m black . --check
	$(PYTHON) -m flake8 .

# Información del proyecto
info:  ## Mostrar información del proyecto
	@echo "📦 Federated Fog Computing Demo"
	@echo "🏗️  Arquitectura: Fog Computing + Federated Learning"
	@echo "🐍 Python: $(shell $(PYTHON) --version)"
	@echo "📋 Dependencias principales:"
	@echo "   - PyTorch (Deep Learning)"
	@echo "   - Flower (Federated Learning)"
	@echo "   - Paho MQTT (Fog Communication)"
	@echo "🧪 Tests: $(shell find tests -name "test_*.py" | wc -l) archivos de test"
	@echo "📊 Código: $(shell find . -name "*.py" -not -path "./.venv/*" | wc -l) archivos Python"
