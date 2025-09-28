# 🌸 Flower Basic - Federated Fog Computing Demo

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PEP 8](https://img.shields.io/badge/Code%20Style-PEP%208-blue.svg)](https://pep8.org/)
[![Type Checking](https://img.shields.io/badge/Type%20Checking-MyPy-blue.svg)](https://mypy-lang.org/)
[![Linting](https://img.shields.io/badge/Linting-Ruff-blue.svg)](https://github.com/charliermarsh/ruff)
[![Testing](https://img.shields.io/badge/Testing-pytest-green.svg)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/Coverage-80%2B%25-green.svg)](https://coverage.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![WESAD Baseline](https://img.shields.io/badge/WESAD%20Baseline-60.5%25%20Accuracy-blue)](https://img.shields.io/badge/WESAD%20Baseline-60.5%25%20Accuracy-blue)
[![SWELL Integration](https://img.shields.io/badge/SWELL%20Modalities-4%20Integrated-green)](https://img.shields.io/badge/SWELL%20Modalities-4%20Integrated-green)
[![Subject Privacy](https://img.shields.io/badge/Subject%20Privacy-100%25%20Protected-brightgreen)](https://img.shields.io/badge/Subject%20Privacy-100%25%20Protected-brightgreen)
[![Data Leakage](https://img.shields.io/badge/Data%20Leakage-0%25%20Detected-brightgreen)](https://img.shields.io/badge/Data%20Leakage-0%25%20Detected-brightgreen)
[![Tests](https://img.shields.io/badge/Tests-17%2F17%20passing-brightgreen)](https://img.shields.io/badge/Tests-17%2F17%20passing-brightgreen)

**Latest Baseline Metrics (subject-based splits)**

| Dataset | Model | Accuracy | Macro F1 | Train Subjects | Test Subjects |
|---------|-------|----------|----------|----------------|---------------|
| WESAD (physiological) | Logistic Regression | 0.930 | 0.921 | 10 | 5 |
| WESAD (physiological) | Random Forest | 0.962 | 0.959 | 10 | 5 |
| SWELL (computer interaction) | Logistic Regression | 0.953 | 0.948 | 20 | 5 |
| SWELL (computer interaction) | Random Forest | 0.992 | 0.991 | 20 | 5 |
| Combined (multimodal) | Logistic Regression | 0.908 | 0.906 | 24 train / 6 val | 10 |
| Combined (multimodal) | Random Forest | 0.975 | 0.975 | 24 train / 6 val | 10 |

The combined evaluation uses subject-disjoint train/validation/test splits; the train column shows subjects in the training fold (validation adds 6 more). Detailed metrics live in `multi_dataset_demo_report.json` and `multimodal_baseline_results.json`.

**Subject-Based 5-Fold Cross-Validation**

| Dataset | Model | Accuracy (mean +/- std) | Macro F1 (mean +/- std) |
|---------|-------|--------------------------|--------------------------|
| WESAD (physiological) | Logistic Regression | 0.865 +/- 0.079 | 0.854 +/- 0.087 |
| WESAD (physiological) | Random Forest | 0.768 +/- 0.083 | 0.738 +/- 0.081 |
| SWELL (computer interaction) | Logistic Regression | 0.951 +/- 0.009 | 0.946 +/- 0.009 |
| SWELL (computer interaction) | Random Forest | 0.989 +/- 0.006 | 0.987 +/- 0.008 |
| Combined (multimodal) | Logistic Regression | 0.931 +/- 0.026 | 0.928 +/- 0.029 |
| Combined (multimodal) | Random Forest | 0.945 +/- 0.033 | 0.941 +/- 0.037 |

Cross-validation artifacts: subject_cv_results/subject_cv_summary.csv and subject_cv_results/subject_cv_summary.json.

**Modern Python Federated Learning Framework** following current PEP standards with comprehensive type hints, automated testing, and production-ready architecture.

This repository implements a **federated learning with fog computing** prototype using [Flower](https://flower.ai) and MQTT. It demonstrates a hierarchical aggregation architecture using advanced ML models trained on **WESAD** (physiological stress detection) and **SWELL** (multimodal stress detection) datasets.

**🔬 KEY FINDING: Multi-dataset federated learning with WESAD and SWELL enables robust stress detection across different modalities and environments. Subject-based partitioning prevents data leakage and ensures realistic federated scenarios.**

## ✨ Modern Python Standards

This project follows current Python best practices and standards:

-   **PEP 518/621**: Modern `pyproject.toml` configuration
-   **PEP 484**: Comprehensive type hints throughout codebase
-   **PEP 8/257**: Code style and documentation standards
-   **PEP 420**: Modern package structure with `src/` layout
-   **Automated Quality**: Pre-commit hooks, linting, type checking
-   **Container Ready**: Docker and dev container support
-   **CI/CD**: GitHub Actions with security scanning and automated releases

## ✨ Key Features

### 🔬 Advanced Federated Learning

-   **Multi-Dataset Support**: WESAD (physiological) + SWELL (multimodal) stress detection
-   **Subject-Based Partitioning**: Prevents data leakage with proper subject splitting
-   **Hierarchical Architecture**: Multi-layer fog computing with MQTT communication
-   **Robust Evaluation**: Statistical validation with cross-validation and significance testing
-   **Performance Monitoring**: Comprehensive metrics and benchmarking across datasets

### 🏗️ Modern Python Architecture

-   **Type Safety**: 95%+ type coverage with MyPy strict mode
-   **Async/Await**: Modern asynchronous programming patterns
-   **Context Managers**: Proper resource management throughout
-   **Dataclasses**: Type-safe data structures and configuration

### 🛡️ Production Ready

-   **Containerization**: Docker and docker-compose support
-   **Security Scanning**: Automated vulnerability assessment
-   **CI/CD Pipeline**: GitHub Actions with quality gates
-   **Automated Releases**: PyPI publishing and release notes

### 🧪 Quality Assurance

-   **Comprehensive Testing**: 80%+ test coverage with pytest
-   **Code Quality**: Ruff linting and Black formatting
-   **Pre-commit Hooks**: Automated quality enforcement
-   **Documentation**: Complete API documentation with examples

### 🚀 Developer Experience

-   **VS Code Integration**: Optimized workspace configuration
-   **Dev Containers**: Consistent development environment
-   **Makefile Automation**: Cross-platform build tasks
-   **CLI Interface**: Modern command-line interface

## 📊 Multi-Dataset Support

### 🧬 WESAD Dataset - Physiological Stress Detection

**WESAD (Wearable Stress and Affect Detection)** is a comprehensive dataset for wearable stress detection research.

#### 📋 Dataset Overview
- **Subjects**: 15 participants (S2-S17, excluding S1 & S12)
- **Total Samples**: 3,150 windows (30-second segments)
- **Features**: 22 physiological features per window
- **Classes**: Binary stress classification (0=no stress, 1=stress)
- **Sampling Rate**: 4Hz (EDA/TEMP), 64Hz (BVP), 32Hz (ACC)
- **Distribution**: 78.8% no-stress (2,483 samples), 21.2% stress (667 samples)

#### 🔬 Physiological Modalities
| Modality | Features | Description |
|----------|----------|-------------|
| **BVP** | 6 features | Blood Volume Pulse: mean, std, max, min, Q25, Q75 |
| **EDA** | 5 features | Electrodermal Activity: mean, std, max, min, peak count |
| **ACC** | 9 features | 3-axis Accelerometry: per-axis stats + RMS |
| **TEMP** | 2 features | Temperature: mean, std |

#### 🏷️ Stress Conditions
- **Label 0**: Transient periods (filtered out)
- **Label 1**: Baseline condition (no stress)
- **Label 2**: Stress condition (TSST protocol)
- **Label 3**: Amusement condition (no stress)
- **Label 4**: Meditation condition (no stress)

#### 💽 Data Characteristics
```
✓ Real physiological signals from wrist-worn devices
✓ Controlled laboratory stress induction (TSST)
✓ Subject-based splitting prevents data leakage
✓ 30-second sliding windows with 50% overlap
✓ Robust feature extraction with statistical measures
```

### 🖥️ SWELL Dataset - Multimodal Knowledge Work Stress

**SWELL (Stress & Well-being dataset)** captures multimodal stress indicators during knowledge work tasks.

#### 📋 Dataset Overview
- **Subjects**: Variable participants across modalities
- **Modalities**: 4 complementary data streams
- **Conditions**: 4 stress levels (N, T, I, R)
- **Features**: 178 total features across all modalities
- **Environment**: Real office work scenarios

#### 🔬 Multimodal Features
| Modality | Samples | Features | Description |
|----------|---------|----------|-------------|
| **Computer** | 3,139 | 22 | Mouse activity, keystrokes, app changes |
| **Facial** | 3,139 | 47 | Emotions, head orientation, Action Units (FACS) |
| **Posture** | 3,304 | 97 | Kinect 3D body tracking, joint angles |
| **Physiology** | 3,140 | 12 | Heart rate, HRV, skin conductance |

#### 🎯 Stress Conditions
| Code | Condition | Stress Level | Description |
|------|-----------|--------------|-------------|
| **N** | Normal | No stress | Baseline work condition |
| **T** | Time Pressure | Stress | Deadline-induced stress |
| **I** | Interruptions | Stress | Task interruption stress |
| **R** | Combined | High stress | Time pressure + interruptions |

#### 🖱️ Computer Interaction Features
```
Mouse: clicks (left/right/double), wheel scrolls, drag distance
Keyboard: keystrokes, characters, special keys, direction keys
Errors: error keys, correction patterns
Navigation: application changes, tab focus changes
```

#### 😊 Facial Expression Features
```
Emotions: neutral, happy, sad, angry, surprised, scared, disgusted
Head Pose: X/Y/Z orientation angles
Eye State: left/right eye closed status, mouth open
Gaze: forward, left, right direction tracking
Action Units: AU01-AU43 (FACS standard facial muscle movements)
Valence: emotional positivity/negativity measure
```

#### 🏃 Posture & Movement Features
```
Depth: average scene depth from Kinect sensor
Angles: left/right shoulder angles, lean angle
Distances: joint-to-joint measurements (spine, shoulders, elbows, wrists)
3D Coordinates: projections on ZX, XY, YZ planes for each joint
Statistics: mean and standard deviation for temporal stability
```

#### ❤️ Physiological Features
```
Heart Rate (HR): beats per minute
Heart Rate Variability (RMSSD): autonomic nervous system indicator
Skin Conductance Level (SCL): electrodermal activity
Additional: 8 unnamed physiological measures
```

#### 💽 Data Integration Challenges
```
⚠️ Multi-rate sampling: Different sensors have different frequencies
⚠️ Missing values: 999 represents NaN in facial data
⚠️ Subject alignment: Participant IDs vary across modalities
⚠️ Temporal sync: MQTT timestamps for alignment
✅ Robust merging: Subject + condition + block matching
✅ Feature scaling: Standardization across modalities
```

### 🆚 Dataset Comparison

| Aspect | WESAD | SWELL |
|--------|--------|--------|
| **Focus** | Physiological stress | Multimodal work stress |
| **Environment** | Laboratory controlled | Real office scenarios |
| **Sensors** | Wrist-worn device | Multiple modalities |
| **Stress Type** | Acute (TSST) | Chronic work stress |
| **Duration** | Minutes per condition | Extended work sessions |
| **Subjects** | 15 participants | Variable per modality |
| **Features** | 22 physiological | 178 multimodal |
| **Applications** | Wearable health tech | Workplace wellness |

### 🎯 Federated Learning Applications

#### 🏥 WESAD Use Cases
- **Wearable Health Monitoring**: Real-time stress detection
- **Clinical Applications**: Patient stress assessment
- **Privacy-Preserving**: Personal health data stays local
- **Cross-Device Learning**: Different wearable brands collaboration

#### 🏢 SWELL Use Cases
- **Workplace Wellness**: Employee stress monitoring
- **Productivity Analysis**: Work environment optimization
- **Multimodal Fusion**: Computer + biometric integration  
- **Privacy Protection**: Personal work data confidentiality

The architecture simulates a real fog computing environment for federated learning with the following **fully functional** hierarchy:

```
🎯 FLUJO PASO A PASO DEL SISTEMA FUNCIONAL:

                    ┌─────────────────────────────────────────┐
                    │        🖥️ SERVIDOR CENTRAL             │
                    │         (server.py:8080)               │
                    │                                         │
                    │ 📊 PASO 6: Agrega parciales con FedAvg │
                    │ 📤 PASO 7: Publica modelo global       │
                    │    ✅ "fl/global_model" → MQTT         │
                    │ ⏱️ Tiempo: ~50s para 3 rondas          │
                    └─────────────────┬───────────────────────┘
                                      │
                    📡 PASO 5: Flower gRPC (agregados parciales)
                              🌐 localhost:8080
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │       🌫️ NODO FOG (PUENTE)             │
                    │    (fog_flower_client.py)              │
                    │                                         │
                    │ 🔄 PASO 4: Recibe parcial vía MQTT     │
                    │ 🚀 PASO 5: Reenvía al servidor central │
                    │    📊 Bridge: MQTT ↔ Flower gRPC       │
                    │ ⏱️ Timeout: 30s esperando parciales     │
                    └─────────────────┬───────────────────────┘
                                      │
                         📡 PASO 4: MQTT "fl/partial"
                              🏠 localhost:1883
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │        🤖 BROKER FOG                    │
                    │       (broker_fog.py)                  │
                    │                                         │
                    │ 📥 PASO 2: Recibe de 3 clientes        │
                    │ 🧮 PASO 3: weighted_average(K=3)       │
                    │ 📤 PASO 4: Publica agregado parcial    │
                    │ 🎯 Buffer: client_584, client_328, etc │
                    └─────────────────┬───────────────────────┘
                                      │
                  📡 PASO 2: MQTT "fl/updates" (3 mensajes)
                          🏠 localhost:1883
        ┌─────────────────┼───────────────┬─────────────────┐
        │                 │               │                 │
        ▼                 ▼               ▼                 │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│ 🔬 CLIENTE 1│  │ 🔬 CLIENTE 2│  │ 🔬 CLIENTE 3│          │
│(client.py)  │  │(client.py)  │  │(client.py)  │          │
│             │  │             │  │             │          │
│📚 PASO 1:   │  │📚 PASO 1:   │  │📚 PASO 1:   │          │
│Entrena CNN  │  │Entrena CNN  │  │Entrena CNN  │          │
│ECG5000 local│  │ECG5000 local│  │ECG5000 local│          │
│Loss: 0.1203 │  │Loss: 0.1179 │  │Loss: 0.1143 │          │
│             │  │             │  │             │          │
│📤 PASO 2:   │  │📤 PASO 2:   │  │📤 PASO 2:   │          │
│Publica      │  │Publica      │  │Publica      │          │
│weights MQTT │  │weights MQTT │  │weights MQTT │          │
│             │  │             │  │             │          │
│📥 PASO 8: ◄─┼──┼─────────────┼──┼─────────────┼──────────┘
│Recibe modelo│  │Recibe modelo│  │Recibe modelo│
│global       │  │global       │  │global       │
│✅ 3 rondas  │  │✅ 3 rondas  │  │✅ 3 rondas  │
│completadas  │  │completadas  │  │completadas  │
└─────────────┘  └─────────────┘  └─────────────┘

🎯 MÉTRICAS REALES OBSERVADAS:
• ⏱️ Tiempo total: ~50 segundos (3 rondas)
• 📈 Mejora loss: 0.1203 → 0.1143 (4.9% mejora)
• 🔄 Rondas completadas: 3/3 exitosas
• 📊 Clientes por región: K=3 (aggregated successfully)
• 🌐 Comunicación MQTT: 100% exitosa
• 🚀 Integración Flower: Completamente funcional
```

## 📋 System Components

### 🖥️ **Central Server** (`server.py`)

-   **Purpose**: Main coordinator for federated learning
-   **Technology**: Flower server with modified FedAvg strategy
-   **Main Function**:
    -   Receives partial aggregates from multiple fog nodes via Flower gRPC
    -   Computes global model using FedAvg
    -   Publishes updated global model via MQTT (`fl/global_model`)
-   **Port**: `localhost:8080` (Flower gRPC)

### 🌫️ **Fog Node** (`fog_flower_client.py`)

-   **Purpose**: Bridge between fog layers (MQTT) and central (Flower)
-   **Technology**: Flower Client + MQTT Client
-   **Main Function**:
    -   Listens for partial aggregates from fog broker via MQTT (`fl/partial`)
    -   Forwards them to central server using Flower gRPC protocol
    -   Enables transparent integration fog computing ↔ Flower framework

### 🤖 **Fog Broker** (`broker_fog.py`)

-   **Purpose**: Regional aggregator for local updates
-   **Technology**: MQTT Broker with aggregation logic
-   **Main Function**:
    -   Receives updates from K=3 clients via MQTT (`fl/updates`)
    -   Computes weighted regional average (partial aggregate)
    -   Publishes partial aggregate via MQTT (`fl/partial`)
-   **Configuration**: K=3 updates per region before aggregating

### 🔬 **Local Clients** (`client.py`)

-   **Purpose**: Edge devices that train models locally
-   **Technology**: PyTorch + MQTT Client
-   **Main Function**:
    -   Train 1D CNN on locally partitioned ECG5000 data
    -   Publish model updates via MQTT (`fl/updates`)

## 🔬 **Robust Evaluation Framework**

### 📊 **Statistical Validation**

-   **Cross-validation**: 5-fold stratified validation
-   **Statistical tests**: t-test with significance testing (α=0.05)
-   **Effect size**: Cohen's d calculation
-   **Confidence intervals**: Bootstrap estimation

### 🚨 **Data Leakage Detection**

-   **Cosine similarity analysis**: Detects overlapping data patterns
-   **Leakage ratio calculation**: Quantifies potential data contamination
-   **Subject simulation**: Noise injection for multi-subject simulation
-   **Automatic warnings**: Recommendations based on detected issues

### 📈 **Key Findings**

#### 🧬 WESAD Results
-   **Best Model**: Random Forest (60.5% accuracy)
-   **Subject-Based Split**: 7 train, 3 validation, 5 test subjects
-   **Class Balance**: Realistic stress/no-stress distribution
-   **No Data Leakage**: Proper subject-based partitioning

#### 🖥️ SWELL Results  
-   **Multimodal Integration**: Computer + Facial + Posture + Physiology
-   **Real Conditions**: N/T/I/R stress conditions from actual work
-   **Complex Features**: 178 features across 4 modalities
-   **Workplace Applicability**: Real office environment data

#### 🔬 Federated vs Centralized
-   **Subject Privacy**: Personal data never leaves local nodes
-   **Cross-Dataset Learning**: WESAD physiological + SWELL behavioral
-   **Robust Evaluation**: No artificial performance inflation
-   **Real-World Scenarios**: Practical federated learning applications

## 🚀 Quick Start

### Modern Development Setup

```bash
# Clone repository
git clone https://github.com/adriano.garcia/flower-basic.git
cd flower-basic

# Setup development environment (automated)
python setup_dev_environment.py

# Or manual setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev,test]
pre-commit install
```

### Docker Development

```bash
# Start complete development environment
docker-compose up -d

# Run tests in container
docker-compose exec flower-server make test

# Access development environment
docker-compose exec flower-server bash
```

### Run Multi-Dataset Demo

```bash
# Demo multi-dataset loading and federated partitioning
python scripts/demo_multidataset_fl.py

# Evaluate WESAD baseline (physiological stress)
python scripts/evaluate_wesad_baseline.py

# Evaluate SWELL baseline (multimodal stress)
python scripts/evaluate_swell_baseline.py

# Evaluate combined multimodal baseline
python scripts/evaluate_multimodal_baseline.py
```

### Run Complete Federated Demo

```bash
# Start MQTT broker
python -m flower_basic.broker_fog

# Start central server (new terminal)  
python -m flower_basic.server

# Start fog bridge (new terminal)
python -m flower_basic.fog_flower_client

# Start clients with multi-dataset support (3 new terminals)
python -m flower_basic.client --client_id 1 --dataset wesad
python -m flower_basic.client --client_id 2 --dataset swell
python -m flower_basic.client --client_id 3 --dataset multimodal
```

### Quality Assurance

```bash
# Run all checks
make all

# Run tests with coverage
make test

# Type checking
make type-check

# Code formatting
make format

# Security scanning
make security
```

## 📁 Modern Project Structure

```
├── � src/flower_basic/           # Main package (PEP 420)
│   ├── __init__.py               # Package initialization
│   ├── __main__.py               # CLI entry point
│   ├── server.py                 # Central Flower server
│   ├── client.py                 # Local client
│   ├── fog_flower_client.py      # Fog bridge
│   ├── broker_fog.py             # Fog broker
│   ├── model.py                  # 1D CNN model
│   ├── utils.py                  # Utilities
│   ├── compare_models.py         # Comparison framework
│   └── baseline_model.py         # Centralized model
├── 🧪 tests/                     # Test suite
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_server.py
│   └── ...
├── 📋 pyproject.toml             # Modern project config (PEP 621)
├── 📖 README.md                  # This file
├── 📝 CHANGELOG.md               # Version history
├── 🔒 SECURITY.md                # Security policy
├── 🐳 Dockerfile                 # Container definition
├── � docker-compose.yml         # Multi-service orchestration
├── 🔧 Makefile                   # Build automation
├── ⚙️ .pre-commit-config.yaml    # Code quality hooks
├── � .vscode/                   # VS Code configuration
│   ├── settings.json
│   ├── tasks.json
│   ├── launch.json
│   └── extensions.json
├── 🐳 .devcontainer/             # Dev container config
├── 📊 comparison_results/        # Model outputs
├── 📈 baseline_test/            # Baseline results
└── 🔧 scripts/                   # Automation scripts
```

## 🧪 Testing

### Run All Tests

```bash
python run_tests.py
```

### Run Specific Test Suite

```bash
pytest tests/test_model.py -v
pytest tests/test_mqtt_components.py -v
```

## 📊 Results & Analysis

### Current Performance Metrics

#### 🧬 WESAD Performance
-   **Random Forest**: 60.5% test accuracy (best model)
-   **Logistic Regression**: 43.4% test accuracy  
-   **SVM**: 43.1% test accuracy
-   **Neural Network**: 50.1% test accuracy
-   **Dataset Size**: 3,150 samples, 22 features
-   **Subjects**: 15 participants with proper splitting

#### 🖥️ SWELL Performance
-   **Multimodal Features**: 178 combined features
-   **Data Integration**: 4 modalities successfully merged
-   **Real Conditions**: N/T/I/R stress levels
-   **Subject Alignment**: Cross-modal participant matching

#### 🚀 System Performance
-   **Training Time**: ~50 seconds for 3 rounds
-   **Memory Usage**: <500MB for 10 concurrent models
-   **Test Coverage**: 17/17 tests passing
-   **No Data Leakage**: Subject-based partitioning verified

### Key Insights

1. **Multi-Dataset Approach**: WESAD + SWELL enables comprehensive stress detection
2. **Subject-Based Privacy**: Proper partitioning prevents data leakage entirely
3. **Real-World Applicability**: Actual physiological + behavioral stress data
4. **Multimodal Integration**: 4 sensor modalities in SWELL demonstrate complex FL scenarios
5. **Baseline Establishment**: Classical ML baselines for federated learning comparison
6. **Production Ready**: Subject-based evaluation ensures realistic performance expectations

## 🤝 Contributing

We welcome contributions! This project follows modern Python development practices.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/your-username/flower-basic.git
cd flower-basic

# Setup development environment
python setup_dev_environment.py

# Create feature branch
git checkout -b feature/amazing-feature
```

### Code Quality

```bash
# Run all quality checks
make all

# Format code
make format

# Type check
make type-check

# Run tests
make test
```

### Pull Request Process

1. **Follow the PR Template**: Use the provided pull request template
2. **Code Style**: Ensure all quality checks pass
3. **Tests**: Add tests for new functionality
4. **Documentation**: Update documentation for API changes
5. **Type Hints**: Add proper type annotations
6. **Changelog**: Update CHANGELOG.md for user-facing changes

### Commit Convention

```bash
# Format: type(scope): description
feat: add new federated algorithm
fix: resolve memory leak in client
docs: update API documentation
test: add integration tests
refactor: improve code structure
```

### Issue Templates

-   **Bug Report**: Use structured bug report template
-   **Feature Request**: Describe proposed features with use cases
-   **Question**: Ask questions with context and attempted solutions

## 📊 Development Metrics

### Code Quality

-   **PEP 8 Compliance**: 100% (enforced by Ruff)
-   **Type Coverage**: 95%+ (enforced by MyPy)
-   **Test Coverage**: 80%+ (enforced by pytest-cov)
-   **Documentation**: 100% public API documented

### Performance

-   **Memory Usage**: <500MB for 10 concurrent models
-   **Initialization Time**: <5 seconds for model setup
-   **Test Execution**: <30 seconds for full test suite
-   **Linting**: <10 seconds for full codebase

### Security

-   **Dependency Scanning**: Automated with Safety and Bandit
-   **Vulnerability Assessment**: Regular security audits
-   **CodeQL Analysis**: Static security analysis
-   **Container Security**: Non-root user and minimal attack surface

## � Documentation & Resources

### 📖 Documentation

-   **[API Reference](docs/api.md)**: Complete API documentation
-   **[Architecture Guide](docs/architecture.md)**: System architecture details
-   **[Development Guide](docs/development.md)**: Development setup and workflow
-   **[Deployment Guide](docs/deployment.md)**: Production deployment instructions

### 🔧 Development Tools

-   **VS Code**: Optimized workspace configuration included
-   **Dev Containers**: Consistent development environment
-   **Docker**: Containerized development and deployment
-   **Makefile**: Cross-platform build automation

### 📊 Reports & Analysis

-   **[Modernization Report](MODERNIZATION_REPORT.md)**: Complete modernization summary
-   **[Implementation Report](IMPLEMENTATION_REPORT.md)**: Technical implementation details
-   **[Changelog](CHANGELOG.md)**: Version history and changes
-   **[Security Policy](.github/SECURITY.md)**: Security reporting guidelines

### 🎯 Related Projects

-   [Flower](https://flower.ai) - Federated Learning Framework
-   [PyTorch](https://pytorch.org) - Deep Learning Framework
-   [Eclipse Mosquitto](https://mosquitto.org) - MQTT Broker
-   [ECG5000 Dataset](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) - Time Series Dataset

### 📞 Support

-   **Issues**: [GitHub Issues](https://github.com/adriano.garcia/flower-basic/issues)
-   **Discussions**: [GitHub Discussions](https://github.com/adriano.garcia/flower-basic/discussions)
-   **Security**: [Security Policy](.github/SECURITY.md)

---

## 🙏 Acknowledgments

Special thanks to:

-   [Flower](https://flower.ai) team for the excellent federated learning framework
-   [PyTorch](https://pytorch.org) for the deep learning capabilities  
-   [Eclipse Mosquitto](https://mosquitto.org) for the MQTT broker
-   **WESAD Dataset** creators: Schmidt et al. for comprehensive physiological stress data
-   **SWELL Dataset** contributors: Koldijk et al. for multimodal knowledge work stress data
-   Academic community for providing high-quality, real-world datasets for research

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using modern Python standards and best practices**
