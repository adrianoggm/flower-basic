# GitHub Actions CI/CD Pipeline

Este directorio contiene los workflows de GitHub Actions para automatizar testing, quality assurance y deployment del proyecto de federated learning con fog computing.

## 📋 Workflows Disponibles

### 1. **CI - Tests and Code Quality** (`ci.yml`)
**Trigger:** Push y Pull Requests a `main`, `develop`, `task/*`

**Características:**
- ✅ Tests en múltiples versiones de Python (3.10, 3.11)
- ✅ Verificación de formato de código (Black, isort)
- ✅ Linting con flake8
- ✅ Type checking con mypy
- ✅ Coverage de tests con Codecov
- ✅ Tests de integración con MQTT

### 2. **PR - Code Review and Testing** (`pr-review.yml`) 
**Trigger:** Pull Requests

**Características:**
- 🔍 Review automático de código
- 📊 Comentarios automáticos en PRs con resultados
- 🧪 Tests de compatibilidad multi-versión
- 🔒 Análisis de seguridad con Trivy
- ⚡ Tests de regresión de performance
- 📋 Resumen ejecutivo en PR

### 3. **Release and Deployment** (`release.yml`)
**Trigger:** Releases y manual dispatch

**Características:**
- 🚀 Quality gates para releases
- 🛡️ Análisis de seguridad (Bandit, Safety)
- 📦 Creación de artefactos de deployment
- ⚡ Tests de performance
- 🎯 Deployment automático con validación

### 4. **Nightly Tests** (`nightly.yml`)
**Trigger:** Diario a las 2 AM UTC y manual

**Características:**
- 🌙 Suite de tests extendida
- 🔍 Análisis profundo de calidad de código
- 🛡️ Auditoría de seguridad completa
- 📊 Monitoreo de dependencias
- 🚨 Creación automática de issues en fallos

## 🚀 Uso Rápido

### Comandos Locales (Compatibles con CI)
```bash
# Formateo (igual que en CI)
black --check --diff .
isort --check-only --diff .

# Tests (igual que en CI)
pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

# Linting (igual que en CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Verificación completa
python format_code.py --check
```

### Manual Triggers
```bash
# Trigger release workflow
gh workflow run release.yml -f environment=staging

# Trigger nightly tests
gh workflow run nightly.yml -f test_scope=full

# Trigger PR review (automático en PRs)
# Se ejecuta automáticamente en cada PR
```

## 📊 Badges para README

Agrega estos badges al README principal:

```markdown
[![CI](https://github.com/adrianoggm/flower-basic/workflows/CI%20-%20Tests%20and%20Code%20Quality/badge.svg)](https://github.com/adrianoggm/flower-basic/actions/workflows/ci.yml)
[![Release](https://github.com/adrianoggm/flower-basic/workflows/Release%20and%20Deployment/badge.svg)](https://github.com/adrianoggm/flower-basic/actions/workflows/release.yml)
[![Nightly](https://github.com/adrianoggm/flower-basic/workflows/Nightly%20Tests%20-%20Extended%20Testing%20Suite/badge.svg)](https://github.com/adrianoggm/flower-basic/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/adrianoggm/flower-basic/branch/main/graph/badge.svg)](https://codecov.io/gh/adrianoggm/flower-basic)
```

## 🔧 Configuración de Secrets

Para funcionalidad completa, configura estos secrets en GitHub:

```bash
# Para Codecov (opcional)
CODECOV_TOKEN=<tu-token-codecov>

# Para deployment (si usas servicios externos)
DEPLOY_TOKEN=<token-deployment>
SLACK_WEBHOOK=<webhook-notificaciones>
```

## 📈 Métricas y Monitoreo

### Quality Gates
- **Test Coverage**: Mínimo 80% (configurable en workflows)
- **Linting**: Sin errores críticos (E9, F63, F7, F82)
- **Security**: Sin vulnerabilidades críticas
- **Performance**: Sin regresiones detectadas

### Artifacts Generados
- **Test Results**: XML y HTML coverage reports
- **Security Reports**: Bandit, Safety, Trivy scans
- **Quality Reports**: Pylint, complexity analysis
- **Deployment Packages**: Tar.gz con versioning

## 🛠️ Troubleshooting

### Tests Failing Localmente
```bash
# Ejecutar tests como en CI
pytest tests/ -v --tb=short

# Verificar formato
black --check .
isort --check-only .

# Ejecutar quality check completo
python format_code.py
```

### Performance Issues
```bash
# Profile memory usage
python -c "
import tracemalloc
tracemalloc.start()
from model import ECGModel
model = ECGModel()
current, peak = tracemalloc.get_traced_memory()
print(f'Memory: {peak / 1024 / 1024:.2f}MB')
"
```

### Security Scans
```bash
# Local security scan
bandit -r . -f json
safety check
```

## 🎯 Mejores Prácticas

1. **Pre-commit**: Los workflows replican checks locales
2. **Branch Protection**: Configurar require status checks
3. **Automatic Merging**: Solo si todos los checks pasan
4. **Artifact Retention**: 30 días para reports, 7 días para tests
5. **Performance Monitoring**: Alertas automáticas en regresiones

## 📚 Referencias

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linting](https://flake8.pycqa.org/)
- [Codecov Integration](https://docs.codecov.com/docs)
