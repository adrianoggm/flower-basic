"""Federated Learning with Fog Computing Demo.

A comprehensive implementation of federated learning using Flower framework
with fog computing architecture for ECG classification.

This package provides:
- Federated learning simulation with fog computing
- Robust evaluation framework with statistical validation
- Data leakage detection and analysis
- Comprehensive testing and benchmarking tools

Example:
    >>> from flower_basic import ModelComparator
    >>> comparator = ModelComparator()
    >>> results = comparator.run_robust_comparison()
"""

__version__ = "0.1.0"
__author__ = "Adriano Garcia"
__email__ = "adriano.garcia@example.com"

from .compare_models import ModelComparator
from .baseline_model import BaselineTrainer
from .model import ECGModel
from .utils import (
    load_ecg5000_subject_based,
    detect_data_leakage,
    statistical_significance_test,
)

__all__ = [
    "ModelComparator",
    "BaselineTrainer",
    "ECGModel",
    "load_ecg5000_subject_based",
    "detect_data_leakage",
    "statistical_significance_test",
]