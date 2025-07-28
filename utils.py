"""Utility functions for federated learning with fog computing."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_ecg5000_openml(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download ECG5000 from OpenML and split into train/test sets.

    Args:
        test_size: Proportion of dataset to use for testing
        random_state: Random seed for reproducible splits

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as NumPy arrays
    """
    # Fetch OpenML "ECG5000" by name
    ds = fetch_openml(name="ECG5000", version=1, as_frame=False)
    X, y = ds["data"], ds["target"].astype(int)

    # Binarize: class 1 → normal (0), else abnormal (1)
    y = (y != 1).astype(int)

    return train_test_split(
        X.astype(np.float32),
        y.astype(np.int64),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def state_dict_to_numpy(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PyTorch state_dict to JSON-serializable dict.

    Converts PyTorch tensors and numpy arrays to lists for JSON serialization.
    This is used for MQTT transmission of model parameters.

    Args:
        state_dict: Dictionary containing model parameters

    Returns:
        Dictionary with parameters converted to JSON-serializable lists
    """
    np_dict = {}
    for k, v in state_dict.items():
        # Handle both PyTorch tensors and numpy arrays
        if hasattr(v, "detach"):
            # PyTorch tensor
            np_dict[k] = v.detach().cpu().numpy().tolist()
        elif hasattr(v, "tolist"):
            # numpy array
            np_dict[k] = v.tolist()
        else:
            # Already a list or other serializable type
            np_dict[k] = v
    return np_dict


def numpy_to_state_dict(
    np_dict: Dict[str, Any], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Convert JSON-loaded dict back to PyTorch state_dict.

    Converts lists back to PyTorch tensors for model loading.
    This is used after receiving model parameters via MQTT.

    Args:
        np_dict: Dictionary with parameters as lists
        device: Target device for tensors (CPU if None)

    Returns:
        Dictionary mapping parameter names to PyTorch tensors
    """
    state_dict = {}
    for k, v in np_dict.items():
        tensor = torch.tensor(v, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        state_dict[k] = tensor
    return state_dict
