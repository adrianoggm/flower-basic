# 📋 Development Rules & Standards

**Modern Python Development Standards for Flower-Basic Project**

This document defines the development standards, code quality requirements, and best practices for contributing to the Flower-Basic federated learning framework.

## 🎯 **Project Vision & Standards**

This project follows **modern Python development best practices** with emphasis on:
- **Type Safety**: Comprehensive type hints and MyPy strict mode
- **Code Quality**: Automated formatting, linting, and testing
- **Documentation**: Complete API documentation and examples
- **Security**: Automated vulnerability scanning and secure coding practices
- **Performance**: Optimized algorithms and resource management
- **Maintainability**: Clean architecture and modular design

---

## 📚 **Python Standards Compliance**

### **PEP Standards (Required)**

| PEP | Title | Implementation | Verification |
|-----|-------|----------------|--------------|
| **PEP 8** | Style Guide for Python Code | Black formatter (line length: 88) | `black --check .` |
| **PEP 257** | Docstring Conventions | Google/NumPy style docstrings | Manual review + linting |
| **PEP 484** | Type Hints | Comprehensive type annotations | `mypy --strict .` |
| **PEP 518** | Build System Requirements | Modern `pyproject.toml` config | Build system validation |
| **PEP 621** | Project Metadata | Standardized project config | `pip install -e .` validation |
| **PEP 420** | Implicit Namespace Packages | `src/` layout structure | Import testing |

### **Modern Python Features (Required)**

- **Python 3.8+**: Minimum supported version
- **Type Hints**: 95%+ coverage required
- **Dataclasses**: For structured data objects
- **Context Managers**: Proper resource management
- **Async/Await**: Asynchronous programming patterns
- **Pathlib**: Modern path handling
- **f-strings**: Modern string formatting

---

## 🔧 **Code Quality Standards**

### **Formatting & Style**

```yaml
# Enforced by Black formatter
line_length: 88
target_version: ['py38', 'py39', 'py310', 'py311']
skip_string_normalization: false
```

```yaml
# Import sorting with isort
profile: "black"
multi_line_output: 3
line_length: 88
known_first_party: ["flower_basic"]
force_sort_within_sections: true
```

### **Linting Requirements**

```yaml
# Ruff linter configuration
target_version: "py38"
line_length: 88
select: [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings  
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "S",   # bandit security
    "T20", # flake8-print
]
```

### **Type Checking Standards**

```yaml
# MyPy strict configuration
python_version: "3.8"
strict: true
warn_return_any: true
warn_unused_configs: true
disallow_untyped_defs: true
disallow_incomplete_defs: true
check_untyped_defs: true
disallow_untyped_decorators: true
no_implicit_optional: true
warn_redundant_casts: true
warn_unused_ignores: true
show_error_codes: true
```

**Required Type Coverage**: **95%+ minimum**

---

## 📝 **Documentation Standards**

### **Docstring Requirements**

**All public functions, classes, and modules MUST have comprehensive docstrings.**

```python
def load_wesad_dataset(
    subjects: Optional[List[str]] = None,
    signals: List[str] = ['BVP', 'EDA', 'ACC'],
    test_size: float = 0.2,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess WESAD dataset for federated learning.
    
    The WESAD (Wearable Stress and Affect Detection) dataset contains
    physiological signals from 15 subjects collected during different
    stress conditions using Empatica E4 wearable devices.
    
    Args:
        subjects: List of subject IDs to include. If None, includes all subjects.
            Valid subjects: ['S2', 'S3', ..., 'S17'] (excluding S1, S12).
        signals: List of physiological signals to extract.
            Available: ['BVP', 'EDA', 'ACC', 'TEMP', 'HR', 'IBI'].
        test_size: Fraction of data to use for testing (0.0-1.0).
        stratify: Whether to maintain class distribution in train/test split.
    
    Returns:
        A tuple containing:
        - X_train: Training features (n_samples, n_features)
        - X_test: Test features (n_samples, n_features) 
        - y_train: Training labels (n_samples,)
        - y_test: Test labels (n_samples,)
    
    Raises:
        FileNotFoundError: If WESAD data directory is not found.
        ValueError: If invalid subjects or signals are specified.
        DatasetError: If dataset preprocessing fails.
    
    Example:
        ```python
        # Load default configuration
        X_train, X_test, y_train, y_test = load_wesad_dataset()
        
        # Load specific subjects with custom signals
        X_train, X_test, y_train, y_test = load_wesad_dataset(
            subjects=['S2', 'S3', 'S4'],
            signals=['BVP', 'EDA'],
            test_size=0.3
        )
        ```
    
    Note:
        - Data is automatically normalized using StandardScaler
        - Missing values are interpolated using linear interpolation
        - Labels are binary: 0 (no stress), 1 (stress)
        - Original sampling rate: 700Hz (downsampled to 64Hz for efficiency)
    """
```

### **README Documentation**

Each major component MUST include:
- **Purpose**: Clear description of functionality
- **Usage Examples**: Code examples with expected outputs
- **API Reference**: Links to detailed API documentation
- **Performance Metrics**: Benchmarks and resource usage
- **Configuration**: All configurable parameters

---

## 🧪 **Testing Standards**

### **Test Coverage Requirements**

| Component | Minimum Coverage | Test Types Required |
|-----------|------------------|-------------------|
| **Core Logic** | 95% | Unit + Integration |
| **API Endpoints** | 90% | Unit + E2E |
| **Utilities** | 85% | Unit |
| **Scripts** | 80% | Integration |
| **Overall Project** | **85% minimum** | All types |

### **Test Categories & Markers**

```python
# Test markers for pytest
@pytest.mark.unit          # Fast, isolated unit tests
@pytest.mark.integration   # Component integration tests
@pytest.mark.slow          # Long-running tests (>5s)
@pytest.mark.network       # Tests requiring network access
@pytest.mark.security      # Security-focused tests
@pytest.mark.performance   # Performance benchmarking tests
```

### **Test Structure Requirements**

```python
class TestWESADDatasetLoader:
    """Test suite for WESAD dataset loading functionality."""
    
    def test_load_default_configuration(self) -> None:
        """Test loading with default parameters."""
        X_train, X_test, y_train, y_test = load_wesad_dataset()
        
        # Assertions with descriptive messages
        assert X_train.shape[0] > 0, "Training set should not be empty"
        assert X_test.shape[0] > 0, "Test set should not be empty"
        assert X_train.shape[1] == X_test.shape[1], "Feature dimensions must match"
        
        # Type checking
        assert isinstance(X_train, np.ndarray), "Should return numpy arrays"
        assert X_train.dtype == np.float32, "Should use float32 for memory efficiency"
    
    def test_invalid_subjects_raises_error(self) -> None:
        """Test that invalid subject IDs raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid subject ID"):
            load_wesad_dataset(subjects=['S1', 'S99'])  # S1 excluded, S99 doesn't exist
    
    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.5])
    def test_test_size_parameter(self, test_size: float) -> None:
        """Test various test_size configurations."""
        X_train, X_test, y_train, y_test = load_wesad_dataset(test_size=test_size)
        
        total_samples = len(X_train) + len(X_test)
        actual_ratio = len(X_test) / total_samples
        
        assert abs(actual_ratio - test_size) < 0.05, f"Test size ratio should be ~{test_size}"
```

---

## 🚀 **Performance Standards**

### **Resource Requirements**

| Operation | Max Memory | Max Time | Requirements |
|-----------|------------|----------|--------------|
| **Model Training** | 512MB | 30s/epoch | CPU sufficient |
| **Data Loading** | 256MB | 10s | Lazy loading |
| **Client Startup** | 128MB | 5s | Fast initialization |
| **MQTT Communication** | 64MB | 1s/message | Low latency |
| **Aggregation** | 256MB | 5s | Efficient algorithms |

### **Performance Testing**

```python
@pytest.mark.performance
def test_model_training_performance() -> None:
    """Verify model training meets performance requirements."""
    model = ECGModel()
    X_train, _, y_train, _ = load_ecg5000_openml()
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Train for one epoch
    trainer = ModelTrainer(model)
    trainer.train_epoch(X_train, y_train, batch_size=32)
    
    elapsed_time = time.time() - start_time
    peak_memory = psutil.Process().memory_info().rss - start_memory
    
    assert elapsed_time < 30.0, f"Training took {elapsed_time:.2f}s, should be <30s"
    assert peak_memory < 512 * 1024 * 1024, f"Used {peak_memory/1024/1024:.1f}MB, should be <512MB"
```

---

## 🔒 **Security Standards**

### **Security Requirements**

1. **No Hardcoded Secrets**: Use environment variables or secure vaults
2. **Input Validation**: Sanitize all external inputs
3. **Error Handling**: No sensitive data in error messages
4. **Dependencies**: Regular security scanning with automated updates
5. **Network Security**: TLS encryption for all communications

### **Security Testing**

```python
@pytest.mark.security
def test_no_sensitive_data_in_logs() -> None:
    """Ensure sensitive data is not logged."""
    with LogCapture() as log:
        client = FederatedClient(api_key="secret_key_12345")
        client.connect()
        
    # Check that sensitive data is not in logs
    log_messages = str(log)
    assert "secret_key_12345" not in log_messages, "API key found in logs"
    assert "password" not in log_messages, "Potential password in logs"

@pytest.mark.security  
def test_input_sanitization() -> None:
    """Test that malicious inputs are properly sanitized."""
    malicious_inputs = [
        "../../../etc/passwd",
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "{{7*7}}",  # Template injection
    ]
    
    for malicious_input in malicious_inputs:
        with pytest.raises((ValueError, SecurityError)):
            process_user_input(malicious_input)
```

---

## 🔄 **Git Workflow Standards**

### **Branch Naming Convention**

```bash
# Feature branches
feature/wesad-dataset-loader
feature/fog-aware-strategy
feature/node-registry-api

# Bug fixes
fix/mqtt-connection-timeout
fix/memory-leak-client
hotfix/security-vulnerability

# Documentation
docs/api-reference-update
docs/deployment-guide

# Maintenance
chore/dependency-updates
refactor/model-architecture
test/integration-coverage
```

### **Commit Message Format**

```bash
# Format: <type>(<scope>): <description>
#
# <body>
#
# <footer>

feat(dataset): add WESAD dataset loader with comprehensive preprocessing

- Implement subject-based data loading from WESAD dataset
- Add support for multiple physiological signals (BVP, EDA, ACC, TEMP)
- Include data validation and error handling
- Add comprehensive test suite with 95% coverage
- Update documentation with usage examples

Closes #123
References #456
```

**Commit Types**:
- `feat`: New features
- `fix`: Bug fixes  
- `docs`: Documentation updates
- `style`: Code formatting (no logic changes)
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `security`: Security improvements

---

## 📋 **Code Review Standards**

### **Required Review Checklist**

**Functionality** ✅
- [ ] Code solves the stated problem correctly
- [ ] Edge cases are handled appropriately
- [ ] Error handling is comprehensive
- [ ] Performance meets requirements

**Code Quality** ✅
- [ ] Follows PEP 8 style guidelines (automated)
- [ ] Type hints are complete and accurate
- [ ] Functions/classes have single responsibilities
- [ ] Code is readable and well-structured

**Testing** ✅
- [ ] Test coverage meets minimum requirements (85%)
- [ ] Tests cover happy path and edge cases
- [ ] Integration tests verify component interactions
- [ ] Performance tests validate resource usage

**Documentation** ✅
- [ ] Public APIs have comprehensive docstrings
- [ ] README updated for user-facing changes
- [ ] Code comments explain complex logic
- [ ] Examples are provided for new features

**Security** ✅
- [ ] No hardcoded secrets or credentials
- [ ] Input validation prevents injection attacks
- [ ] Dependencies are up-to-date and secure
- [ ] Error messages don't leak sensitive information

### **Review Process**

1. **Automated Checks**: All CI/CD checks must pass
2. **Code Review**: At least one approving review required
3. **Testing**: Manual testing for complex features
4. **Documentation**: Review of user-facing documentation
5. **Security**: Security review for sensitive changes

---

## 🚀 **CI/CD Pipeline Standards**

### **GitHub Actions Workflow**

Our CI/CD pipeline enforces all quality standards automatically:

```yaml
# .github/workflows/ci.yml - Key requirements
jobs:
  test:
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    steps:
      - name: Code Formatting Check
        run: |
          black --check --diff .
          isort --check-only --diff .
      
      - name: Linting
        run: |
          ruff check .
          flake8 . --max-line-length=88
      
      - name: Type Checking
        run: mypy . --strict
      
      - name: Security Scanning
        run: |
          bandit -r src/
          safety check
      
      - name: Testing
        run: |
          pytest tests/ --cov=. --cov-report=xml --cov-fail-under=85
          
      - name: Performance Testing
        run: |
          pytest tests/ -m performance --benchmark-only
```

### **Quality Gates**

**Required for merge**:
- ✅ All tests passing (85%+ coverage)
- ✅ Code formatting (Black + isort)
- ✅ Linting (Ruff + Flake8)  
- ✅ Type checking (MyPy strict)
- ✅ Security scanning (Bandit + Safety)
- ✅ Performance benchmarks
- ✅ At least one code review approval

---

## 🛡️ **Security Guidelines**

### **Dependency Management**

```yaml
# Automated security scanning
- name: Security Audit
  run: |
    # Check for known vulnerabilities
    safety check
    
    # Static security analysis
    bandit -r src/ -f json -o security-report.json
    
    # Dependency license compliance
    pip-licenses --format=json --output-file=licenses.json
```

### **Secure Coding Practices**

```python
# ✅ GOOD: Secure configuration loading
def load_config() -> Config:
    """Load configuration from secure sources."""
    api_key = os.getenv("FL_API_KEY")
    if not api_key:
        raise SecurityError("API key not configured")
    
    return Config(
        api_key=api_key,
        broker_url=os.getenv("MQTT_BROKER_URL", "mqtt://localhost:1883"),
        tls_enabled=os.getenv("TLS_ENABLED", "true").lower() == "true"
    )

# ❌ BAD: Hardcoded secrets
def load_config() -> Config:
    return Config(
        api_key="secret_key_12345",  # Never do this!
        broker_url="mqtt://admin:password@broker:1883"  # Or this!
    )
```

---

## 📊 **Project Structure Standards**

### **Required Directory Structure**

```
flower-basic/
├── 📋 pyproject.toml              # Modern project configuration (PEP 621)
├── 📄 README.md                   # Project documentation
├── 📋 RULES.md                    # This file - development standards
├── 📝 CHANGELOG.md                # Version history
├── 🔒 SECURITY.md                 # Security policy
├── 📄 LICENSE                     # MIT License
├── 🔧 Makefile                    # Cross-platform automation
├── 🐳 Dockerfile                  # Container definition
├── 🐳 docker-compose.yml          # Multi-service orchestration
├── ⚙️ .pre-commit-config.yaml     # Code quality hooks
├── 📁 src/flower_basic/           # Source code (PEP 420)
│   ├── __init__.py               # Package exports
│   ├── __main__.py               # CLI entry point
│   ├── model.py                  # ML models
│   ├── server.py                 # Flower server
│   ├── client.py                 # Federated clients
│   ├── utils.py                  # Utility functions
│   └── datasets/                 # Dataset loaders
│       ├── __init__.py
│       ├── ecg5000.py           # ECG5000 loader
│       ├── wesad.py             # WESAD loader
│       └── swell.py             # SWELL loader
├── 🧪 tests/                     # Test suite
│   ├── __init__.py
│   ├── test_model.py            # Model tests
│   ├── test_datasets.py         # Dataset tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test fixtures
├── 📖 docs/                      # Documentation
│   ├── api/                     # API reference
│   ├── guides/                  # User guides
│   └── examples/                # Code examples
├── 🔧 scripts/                   # Automation scripts
└── ⚙️ .github/                   # GitHub configuration
    ├── workflows/               # CI/CD pipelines
    ├── ISSUE_TEMPLATE/          # Issue templates
    └── PULL_REQUEST_TEMPLATE.md # PR template
```

### **Import Organization**

```python
"""Module docstring describing the module purpose."""

# Standard library imports
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from sklearn.model_selection import train_test_split

# Local imports (relative imports within package)
from flower_basic.model import ECGModel
from flower_basic.utils import get_parameters, set_parameters
from flower_basic.datasets import load_wesad_dataset

# Local imports (specific modules)
from .config import Config
from .exceptions import DatasetError, ModelError
```

---

## ⚡ **Development Workflow**

### **Daily Development Checklist**

**Before Starting Work** ✅
- [ ] Pull latest changes: `git pull origin main`
- [ ] Create feature branch: `git checkout -b feature/my-feature`
- [ ] Verify environment: `make verify-env`

**During Development** ✅
- [ ] Write tests first (TDD approach)
- [ ] Add type hints for all new code
- [ ] Run tests frequently: `make test-watch`
- [ ] Check code quality: `make lint`

**Before Committing** ✅
- [ ] Run full test suite: `make test`
- [ ] Check formatting: `make format-check`
- [ ] Verify type safety: `make type-check`
- [ ] Update documentation if needed

**Before Push** ✅
- [ ] Squash commits if needed
- [ ] Write descriptive commit message
- [ ] Run final checks: `make all`
- [ ] Push and create PR

### **Makefile Targets**

```makefile
# Development commands
.PHONY: all test lint format type-check security docs

all: lint type-check test security    # Run all quality checks

test:                                 # Run test suite
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:                                # Run linting
	ruff check src/ tests/
	flake8 src/ tests/ --max-line-length=88

format:                              # Format code
	black src/ tests/
	isort src/ tests/

format-check:                        # Check formatting
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:                          # Type checking
	mypy src/ --strict

security:                            # Security scanning
	bandit -r src/
	safety check

docs:                               # Generate documentation
	sphinx-build -b html docs/ docs/_build/

install-dev:                        # Install development dependencies
	pip install -e .[dev,test,docs]
	pre-commit install

clean:                              # Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/
```

---

## 🚫 **DEPRECATED COMPONENTS & MIGRATION GUIDELINES**

### **❌ ECG5000 Dataset - DEPRECATED**

**EFFECTIVE DATE: September 27, 2025**

The ECG5000 dataset is **officially deprecated** and should **NOT** be used for new development or research.

**Reasons for deprecation:**
- **Data Leakage Issues**: 92.1% similarity detected between training and test sets
- **Artificial Performance Inflation**: Results are not representative of real-world federated scenarios  
- **Limited Research Value**: Does not provide meaningful insights for federated learning evaluation
- **Poor Subject Separation**: Synthetic subject assignment does not reflect realistic federated data distribution

**Migration Path:**
```python
# ❌ DEPRECATED - Do not use
from flower_basic.datasets import load_ecg5000_dataset
X_train, X_test, y_train, y_test = load_ecg5000_dataset()

# ✅ RECOMMENDED - Use WESAD for stress detection
from flower_basic.datasets import load_wesad_dataset
X_train, X_test, y_train, y_test = load_wesad_dataset(
    signals=['BVP', 'EDA', 'ACC', 'TEMP'],
    conditions=['baseline', 'stress']
)

# ✅ FUTURE - Use SWELL for workload detection (when implemented)
from flower_basic.datasets import load_swell_dataset
X_train, X_test, y_train, y_test = load_swell_dataset(
    workload_levels=['low', 'high'],
    modalities=['EEG', 'ECG']
)
```

**Code Quality Impact:**
- All new code MUST use WESAD or SWELL datasets
- ECG5000-related code will trigger CI/CD warnings
- Pull requests using ECG5000 will be automatically flagged for review

**Timeline:**
- **Phase 1 (Weeks 1-2)**: Migrate existing examples to WESAD
- **Phase 2 (Weeks 3-4)**: Remove ECG5000 from main workflows
- **Phase 3 (Week 5+)**: Complete removal of ECG5000 support

### **📊 Approved Datasets for Development**

| Dataset | Status | Use Case | Subjects | Modalities | Federated Ready |
|---------|---------|----------|----------|------------|-----------------|
| **WESAD** | ✅ **ACTIVE** | Stress Detection | 15 | BVP, EDA, ACC, TEMP | ✅ Yes |
| **SWELL** | 🚧 **IN DEVELOPMENT** | Workload Detection | Multiple | EEG, ECG, Facial | ✅ Yes |
| **ECG5000** | ❌ **DEPRECATED** | ~~Time Series~~ | ~~5000~~ | ~~ECG~~ | ❌ No |

### **Enforcement Rules**

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml - Add ECG5000 detection
repos:
  - repo: local
    hooks:
      - id: check-deprecated-datasets
        name: Check for deprecated ECG5000 usage
        entry: python scripts/check_deprecated.py
        language: python
        files: '\.py$'
```

**CI/CD Integration:**
```yaml
# GitHub Actions - Automated deprecation checking
- name: Check for deprecated ECG5000 usage
  run: |
    if grep -r "ecg5000\|ECG5000" src/ tests/ --exclude-dir=__pycache__; then
      echo "::error::ECG5000 dataset usage detected. Please migrate to WESAD/SWELL."
      exit 1
    fi
```

**Code Review Checklist:**
- [ ] ✅ No ECG5000 dataset usage in new code
- [ ] ✅ WESAD/SWELL datasets used for physiological data
- [ ] ✅ Proper subject-based data partitioning implemented
- [ ] ✅ Data leakage prevention measures in place

## 📋 **Dataset Migration Examples**

### **Example 1: Basic Classification Task**

```python
# ❌ OLD - ECG5000 (deprecated)
def old_federated_training():
    from flower_basic.utils import load_ecg5000_openml
    X_train, X_test, y_train, y_test = load_ecg5000_openml()
    # This has data leakage issues!
    
# ✅ NEW - WESAD (recommended)  
def new_federated_training():
    from flower_basic.datasets import load_wesad_dataset
    X_train, X_test, y_train, y_test = load_wesad_dataset(
        signals=['BVP', 'EDA'],
        conditions=['baseline', 'stress'],
        test_size=0.2,
        normalize=True
    )
    # Proper subject-based splitting, no data leakage
```

### **Example 2: Federated Client Setup**

```python
# ❌ OLD - ECG5000 partitioning (deprecated)
def old_client_setup():
    from flower_basic.datasets.ecg5000 import partition_ecg5000_by_subjects
    # Synthetic subjects, not realistic
    
# ✅ NEW - WESAD partitioning (recommended)
def new_client_setup():
    from flower_basic.datasets.wesad import partition_wesad_by_subjects
    client_datasets = partition_wesad_by_subjects(
        num_clients=5,
        signals=['BVP', 'EDA', 'ACC', 'TEMP'],
        test_size=0.2
    )
    # Real subjects, realistic federated scenario
```

### **Example 3: Model Architecture Updates**

```python
# ❌ OLD - ECG-specific model (deprecated)
class OldECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 32, kernel_size=7)  # ECG-specific
        # Fixed for 140 time points
        
# ✅ NEW - Multi-modal physiological model (recommended)
class PhysiologicalModel(nn.Module):
    def __init__(self, input_dim: int, num_signals: int = 4):
        super().__init__()
        # Flexible architecture for different physiological signals
        self.feature_extractor = nn.ModuleList([
            nn.Conv1d(1, 32, kernel_size=7) for _ in range(num_signals)
        ])
        self.fusion = nn.Linear(32 * num_signals, 64)
        self.classifier = nn.Linear(64, 2)  # Binary classification
```

---

## 📈 **Quality Metrics & Monitoring**

### **Code Quality Dashboard**

| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| **Test Coverage** | ≥85% | 87% | ↗️ |
| **Type Coverage** | ≥95% | 96% | ↗️ |
| **Linting Score** | 9.5/10 | 9.7/10 | ↗️ |
| **Security Score** | A+ | A+ | ➡️ |
| **Performance** | <30s tests | 18s | ↗️ |
| **Dependencies** | 0 vulnerabilities | 0 | ➡️ |

### **Automated Reporting**

```python
# Quality reporting in CI/CD
def generate_quality_report() -> Dict[str, Any]:
    """Generate comprehensive quality report."""
    return {
        "test_coverage": get_test_coverage(),
        "type_coverage": get_type_coverage(), 
        "lint_score": get_lint_score(),
        "security_issues": get_security_issues(),
        "performance_metrics": get_performance_metrics(),
        "dependency_health": get_dependency_health()
    }
```

---

## 🎯 **Enforcement & Compliance**

### **Automated Enforcement**

**Pre-commit Hooks** (Mandatory):
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort  
    hooks:
      - id: isort
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
```

**GitHub Branch Protection**:
- ✅ Require status checks (CI/CD)
- ✅ Require branches to be up to date
- ✅ Require review before merging
- ✅ Dismiss stale reviews
- ✅ Restrict pushes to main branch

### **Manual Review Requirements**

**Architecture Changes**: Senior developer review required
**Security Changes**: Security team review required  
**API Changes**: API design review required
**Performance Changes**: Performance team review required

---

## 📞 **Support & Resources**

### **Getting Help**

**Development Questions**:
- 💬 GitHub Discussions for general questions
- 🐛 GitHub Issues for bugs and feature requests  
- 📧 Email maintainers for security issues

**Documentation**:
- 📖 [API Reference](docs/api.md)
- 🏗️ [Architecture Guide](docs/architecture.md)
- 🚀 [Deployment Guide](docs/deployment.md)
- 🧪 [Testing Guide](docs/testing.md)

### **Useful Commands**

```bash
# Quick development setup
make install-dev

# Run all quality checks
make all

# Watch tests during development
make test-watch

# Update dependencies
make update-deps

# Generate security report
make security-report

# Build documentation locally
make docs-serve
```
# REGLAS ESTRICTAS PARA DATASETS DE IA
## PROHIBICIONES Y POLÍTICAS OBLIGATORIAS

### ⛔ PROHIBICIONES ABSOLUTAS

#### 1. **PROHIBIDO MOCK DATA**
- **NUNCA** generar datos sintéticos/mock para algoritmos de IA
- **NUNCA** usar `np.random`, `fake`, `mock` o datos artificiales para ML/AI
- **SOLO** datos reales de SWELL y WESAD permitidos
- Violación de esta regla = ERROR CRÍTICO

#### 2. **DATASETS AUTORIZADOS**
- **ÚNICAMENTE** SWELL y WESAD
- NO otros datasets sin autorización explícita
- Usar datasets completos y originales

### ✅ POLÍTICAS OBLIGATORIAS

#### 1. **PARA EVALUACIONES** (`evaluate_*.py`)
- **OBLIGATORIO**: Usar datasets COMPLETOS (100% de los datos)
- **OBLIGATORIO**: SWELL completo desde `data/SWELL/`
- **OBLIGATORIO**: WESAD completo desde `data/WESAD/`
- **NO** limitar muestras en evaluaciones de rendimiento
- Objetivo: Obtener métricas reales y precisas

#### 2. **PARA TESTS** (`test_*.py`)
- **PERMITIDO**: Usar muestras pequeñas reales (samples/)
- **OBLIGATORIO**: Las muestras deben ser extractos auténticos
- **PROHIBIDO**: Generar datos de prueba artificiales
- Objetivo: Tests rápidos con datos reales

#### 3. **ESTRUCTURA DE DATOS**
```
data/
├── SWELL/           # Dataset completo para evaluaciones
├── WESAD/           # Dataset completo para evaluaciones  
└── samples/         # Muestras reales para tests
    ├── swell_real_sample.pkl
    └── wesad_real_sample.pkl
```

#### 4. **IMPLEMENTACIÓN**
- **Evaluadores**: Cargar datasets completos obligatoriamente
- **Tests**: Usar fallback a samples/ si dataset completo no disponible
- **Desarrollo**: NUNCA mock data, usar samples reales

### 🎯 CASOS DE USO

| Tipo de Script | Datos a Usar | Tamaño | Propósito |
|---------------|--------------|---------|-----------|
| `evaluate_*.py` | Dataset completo | 100% | Métricas finales |
| `test_*.py` | Muestras reales | ~100-1000 samples | Tests rápidos |
| `debug_*.py` | Muestras reales | Pequeñas | Debugging |
| Federated Learning | Dataset completo | 100% | Entrenamiento real |

### ⚡ ENFORCEMENT

#### Detección de Violaciones:
```python
# PROHIBIDO - Detectar y fallar
FORBIDDEN_PATTERNS = [
    'np.random.randn',
    'np.random.normal', 
    'mock_data',
    'fake_data',
    'synthetic_data',
    'generate_mock',
    'create_fake'
]
```

#### Validación Obligatoria:
- Verificar que datos provienen de archivos reales
- Confirmar rutas a datasets originales
- Rechazar cualquier generación artificial

### 📝 CUMPLIMIENTO
- **Desarrollador**: Responsable de seguir estas reglas
- **Code Review**: Verificar cumplimiento estricto  
- **CI/CD**: Fallar build si se detecta mock data
- **Documentación**: Mantener trazabilidad de datos reales

---
**IMPORTANTE**: Estas reglas garantizan la integridad científica y la validez de los resultados de IA.

---

**This document is living and evolves with the project. Last updated: September 27, 2025**

**Questions or suggestions? Open an issue or discussion on GitHub!** 🚀
