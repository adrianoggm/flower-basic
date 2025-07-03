# utils.py
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def load_ecg5000_openml(
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Descarga ECG5000 desde OpenML y divide en train/test.
    - X_* shape = (n_samples, 140)
    - y_* shape = (n_samples,), valores 1 (normal) o !=1 (anomalía)
    """
    data = fetch_openml(name="ECG5000", version=1, as_frame=False)
    X = data.data.astype(np.float32)
    # Algunos targets vienen como strings; convertimos a int:
    y = data.target.astype(int)
    # Normalizamos etiquetas: 1 → 0 (normal),  others → 1 (anomalía)
    y = np.where(y == 1, 0, 1)
    return train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=y
    )
