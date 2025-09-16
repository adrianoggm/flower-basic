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

from .baseline_model import BaselineTrainer
from .compare_models import ModelComparator
from .model import ECGModel, get_parameters, set_parameters
from .utils import (
    detect_data_leakage,
    load_ecg5000_openml,
    load_ecg5000_subject_based,
    state_dict_to_numpy,
    statistical_significance_test,
)
from .broker_fog import weighted_average

__all__ = [
    "ModelComparator",
    "BaselineTrainer",
    "ECGModel",
    "load_ecg5000_subject_based",
    "detect_data_leakage",
    "statistical_significance_test",
]
