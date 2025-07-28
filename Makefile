# Makefile para Federated Fog Computing Demo
# Automatiza tareas de desarrollo, testing y formatting

PYTHON = .venv/Scripts/python.exe
ifeq ($(OS),Windows_NT)
    PYTHON = .venv/Scripts/python.exe
else
    PYTHON = .venv/bin/python
endif

.PHONY: help install format lint test test-unit test-integration test-coverage clean setup

help:  ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Configurar entorno de desarrollo
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Entorno configurado. Activa con: .venv\Scripts\activate (Windows) o source .venv/bin/activate (Unix)"

install:  ## Instalar dependencias
	$(PYTHON) -m pip install -r requirements.txt

format:  ## Formatear código con Black e isort
	$(PYTHON) -m isort .
	$(PYTHON) -m black .
	@echo "✅ Código formateado"

lint:  ## Ejecutar linting con flake8
	$(PYTHON) -m flake8 .
	@echo "✅ Linting completado"

type-check:  ## Verificar tipos con MyPy
	$(PYTHON) -m mypy . || true
	@echo "✅ Verificación de tipos completada"

test:  ## Ejecutar todos los tests
	$(PYTHON) -m pytest tests/ -v

test-unit:  ## Ejecutar solo tests unitarios
	$(PYTHON) -m pytest tests/ -v -m "not integration and not slow"

test-integration:  ## Ejecutar tests de integración
	$(PYTHON) -m pytest tests/ -v -m integration

test-coverage:  ## Ejecutar tests con reporte de cobertura
	$(PYTHON) -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
	@echo "📊 Reporte de cobertura en htmlcov/index.html"

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
