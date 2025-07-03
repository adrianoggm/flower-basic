# utils.py

import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def load_ecg5000_openml(test_size: float = 0.2, random_state: int = 42):
    """
    Download ECG5000 from OpenML, split into train/test and return
    NumPy arrays X_train, X_test, y_train, y_test.
    """
    # fetch OpenML "ECG5000" by name
    ds = fetch_openml(name="ECG5000", version=1, as_frame=False)
    X, y = ds["data"], ds["target"].astype(int)
    # Binarize: class 1 → normal (0), else abnormal (1)
    y = (y != 1).astype(int)
    return train_test_split(
        X.astype(np.float32), y.astype(np.int64),
        test_size=test_size, random_state=random_state,
        stratify=y
    )

def state_dict_to_numpy(state_dict: dict) -> dict:
    """
    Convert a PyTorch model.state_dict() into a JSON‐serializable
    dict of lists (one list per tensor).
    """
    np_dict = {}
    for k, v in state_dict.items():
        # move to CPU, convert to numpy, then to native Python list
        np_dict[k] = v.detach().cpu().numpy().tolist()
    return np_dict

def numpy_to_state_dict(np_dict: dict, device: torch.device = None) -> dict:
    """
    Convert back from JSON‐loaded dict of lists into a PyTorch
    state_dict (mapping names → torch.Tensor).
    """
    state_dict = {}
    for k, v in np_dict.items():
        tensor = torch.tensor(v, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        state_dict[k] = tensor
    return state_dict
