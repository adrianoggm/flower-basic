# =============================================================================
# utils.py
# =============================================================================
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_data(n_clients=2, samples_per_client=500, features=20, classes=2):
    """
    Generate synthetic classification data and split among n_clients.
    Returns dict: {client_id: (X_train, y_train, X_test, y_test)}
    """
    data = {}
    for cid in range(n_clients):
        X, y = make_classification(
            n_samples=samples_per_client,
            n_features=features,
            n_informative=int(features/2),
            n_redundant=int(features/4),
            n_classes=classes,
            random_state=42 + cid,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        data[cid] = (X_train.astype(np.float32), y_train.astype(np.int64),
                     X_test.astype(np.float32), y_test.astype(np.int64))
    return data
