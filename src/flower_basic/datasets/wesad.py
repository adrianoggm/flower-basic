"""WESAD (Wearable Stress and Affect Detection) dataset loader.

This module provides comprehensive loading and preprocessing for the WESAD dataset,
a multimodal dataset for stress detection using wearable physiological sensors.

The WESAD dataset contains:
- 15 subjects (S2-S17, excluding S1 and S12)
- Physiological signals: BVP, EDA, ACC, TEMP, HR, IBI
- Conditions: baseline, stress, amusement, meditation
- Sampling rates: 700Hz (chest), 64Hz (wrist) for most signals
- Labels: stress detection (binary classification)

Reference:
    Schmidt, P., Reiss, A., Duerr, R., Marberger, C., & Van Laerhoven, K. (2018).
    Introducing WESAD, a multimodal dataset for wearable stress and affect detection.
    In Proceedings of the 20th ACM international conference on multimodal interaction.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class WESADDatasetError(Exception):
    """Raised when WESAD dataset loading or processing fails."""

    pass


# WESAD dataset configuration constants
WESAD_SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
    "S11", "S13", "S14", "S15", "S16", "S17"
]  # S1 and S12 excluded as per dataset documentation

WESAD_SIGNALS = {
    "chest": {
        "ACC": 700,   # 3-axis accelerometer
        "ECG": 700,   # Electrocardiogram  
        "EMG": 700,   # Electromyogram
        "EDA": 700,   # Electrodermal activity
        "TEMP": 700,  # Temperature
        "RESP": 700,  # Respiration
    },
    "wrist": {
        "ACC": 32,    # 3-axis accelerometer
        "BVP": 64,    # Blood volume pulse
        "EDA": 4,     # Electrodermal activity
        "TEMP": 4,    # Temperature
    }
}

WESAD_CONDITIONS = {
    0: "not_defined",
    1: "baseline",      # Relaxed state
    2: "stress",        # Trier Social Stress Test (TSST)
    3: "amusement",     # Funny video clips
    4: "meditation",    # Meditation/recovery
    # Additional transient states exist but are typically excluded
}


def load_wesad_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    subjects: Optional[List[str]] = None,
    signals: List[str] = ["BVP", "EDA", "ACC", "TEMP"],
    sensor_location: str = "wrist",
    conditions: List[str] = ["baseline", "stress"],
    test_size: float = 0.2,
    normalize: bool = True,
    window_size: int = 60,
    overlap: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess WESAD dataset for federated learning.

    The WESAD (Wearable Stress and Affect Detection) dataset contains
    physiological signals from 15 subjects collected during different
    stress conditions using chest-worn RespiBAN and wrist-worn Empatica E4 devices.

    Args:
        data_dir: Path to WESAD dataset directory. If None, uses 'data/WESAD'
            relative to current working directory.
        subjects: List of subject IDs to include. If None, includes all available
            subjects. Valid subjects: ['S2', 'S3', ..., 'S17'] (excluding S1, S12).
        signals: List of physiological signals to extract from the specified sensor.
            For wrist: ['BVP', 'EDA', 'ACC', 'TEMP']
            For chest: ['ACC', 'ECG', 'EMG', 'EDA', 'TEMP', 'RESP']
        sensor_location: Sensor location ('wrist' or 'chest'). Defaults to 'wrist'
            as it's more practical for federated edge devices.
        conditions: List of experimental conditions to include.
            Available: ['baseline', 'stress', 'amusement', 'meditation'].
            Defaults to ['baseline', 'stress'] for binary stress detection.
        test_size: Fraction of data to use for testing (0.0-1.0). Defaults to 0.2.
        normalize: Whether to apply StandardScaler normalization per signal.
            Defaults to True.
        window_size: Size of sliding windows in seconds for feature extraction.
            Defaults to 60 seconds.
        overlap: Overlap between windows (0.0-1.0). Defaults to 0.5 (50% overlap).
        random_state: Random seed for reproducible splits. Defaults to 42.

    Returns:
        Tuple containing:
        - X_train: Training features (n_train_windows, n_features)
        - X_test: Test features (n_test_windows, n_features)
        - y_train: Training labels (n_train_windows,)
        - y_test: Test labels (n_test_windows,)

        Labels:
        - 0: Non-stress (baseline, amusement, meditation)
        - 1: Stress (stress condition)

    Raises:
        WESADDatasetError: If dataset loading or preprocessing fails.
        ValueError: If invalid parameters are provided.
        FileNotFoundError: If WESAD data directory or files are not found.

    Example:
        ```python
        # Basic stress detection with wrist sensor
        X_train, X_test, y_train, y_test = load_wesad_dataset()
        
        # Custom configuration with chest sensor and more conditions
        X_train, X_test, y_train, y_test = load_wesad_dataset(
            subjects=['S2', 'S3', 'S4'],
            signals=['ECG', 'EDA', 'RESP'],
            sensor_location='chest',
            conditions=['baseline', 'stress', 'amusement'],
            window_size=30,
            test_size=0.3
        )
        ```

    Note:
        - Data is preprocessed using sliding windows for time series classification
        - Each window contains aggregated features (mean, std, min, max) per signal
        - Sampling rates vary by sensor and signal type
        - Subject-based splitting ensures no data leakage between train/test sets
        - Missing or corrupted data is automatically handled and logged
    """
    # Validate and set default parameters
    if data_dir is None:
        data_dir = Path("data/WESAD")
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"WESAD data directory not found: {data_dir}")
    
    if subjects is None:
        subjects = WESAD_SUBJECTS.copy()
    
    # Validate subjects
    invalid_subjects = set(subjects) - set(WESAD_SUBJECTS)
    if invalid_subjects:
        raise ValueError(f"Invalid subjects: {invalid_subjects}. "
                        f"Valid subjects: {WESAD_SUBJECTS}")
    
    # Validate sensor location and signals
    if sensor_location not in ["wrist", "chest"]:
        raise ValueError(f"sensor_location must be 'wrist' or 'chest', got {sensor_location}")
    
    available_signals = list(WESAD_SIGNALS[sensor_location].keys())
    invalid_signals = set(signals) - set(available_signals)
    if invalid_signals:
        raise ValueError(f"Invalid signals for {sensor_location}: {invalid_signals}. "
                        f"Available: {available_signals}")
    
    # Validate conditions
    available_conditions = ["baseline", "stress", "amusement", "meditation"]
    invalid_conditions = set(conditions) - set(available_conditions)
    if invalid_conditions:
        raise ValueError(f"Invalid conditions: {invalid_conditions}. "
                        f"Available: {available_conditions}")
    
    # Validate other parameters
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
    
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be between 0.0 and 1.0, got {overlap}")
    
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    try:
        logger.info(f"Loading WESAD dataset from {data_dir}")
        logger.info(f"Subjects: {subjects}, Signals: {signals}, "
                   f"Sensor: {sensor_location}, Conditions: {conditions}")
        
        # Load data for all subjects
        all_features = []
        all_labels = []
        all_subject_ids = []
        
        for subject_id in subjects:
            try:
                subject_features, subject_labels = _load_subject_data(
                    data_dir=data_dir,
                    subject_id=subject_id,
                    signals=signals,
                    sensor_location=sensor_location,
                    conditions=conditions,
                    window_size=window_size,
                    overlap=overlap,
                )
                
                all_features.append(subject_features)
                all_labels.append(subject_labels)
                all_subject_ids.extend([subject_id] * len(subject_features))
                
                logger.info(f"Loaded {subject_id}: {len(subject_features)} windows")
                
            except Exception as e:
                logger.warning(f"Failed to load {subject_id}: {e}")
                continue
        
        if not all_features:
            raise WESADDatasetError("No valid subject data could be loaded")
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        subject_ids = np.array(all_subject_ids)
        
        logger.info(f"Combined data: {X.shape[0]} windows, {X.shape[1]} features")
        logger.info(f"Class distribution - Non-stress: {np.sum(y == 0)}, "
                   f"Stress: {np.sum(y == 1)}")
        
        # Apply normalization if requested
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)
            logger.info("Applied StandardScaler normalization")
        else:
            X = X.astype(np.float32)
        
        y = y.astype(np.int64)
        
        # Subject-based train/test split to prevent data leakage
        unique_subjects = np.unique(subject_ids)
        
        # Split subjects into train and test
        train_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # Cannot stratify by subjects easily
        )
        
        # Create train/test masks
        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        logger.info(f"Subject-based split - Train subjects: {sorted(train_subjects)}")
        logger.info(f"Subject-based split - Test subjects: {sorted(test_subjects)}")
        logger.info(f"Train: {X_train.shape[0]} windows, Test: {X_test.shape[0]} windows")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Failed to load WESAD dataset: {e}")
        raise WESADDatasetError(f"WESAD dataset loading failed: {e}") from e


def _load_subject_data(
    data_dir: Path,
    subject_id: str,
    signals: List[str],
    sensor_location: str,
    conditions: List[str],
    window_size: int,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess data for a single subject.
    
    Args:
        data_dir: Path to WESAD dataset directory.
        subject_id: Subject identifier (e.g., 'S2').
        signals: List of signals to extract.
        sensor_location: 'wrist' or 'chest'.
        conditions: List of experimental conditions.
        window_size: Window size in seconds.
        overlap: Window overlap ratio.
    
    Returns:
        Tuple of (features, labels) for the subject.
    """
    subject_path = data_dir / subject_id / f"{subject_id}.pkl"
    
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject file not found: {subject_path}")
    
    # Load pickled data
    with open(subject_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    
    # Extract signal data and labels
    signal_data = data["signal"][sensor_location]
    labels = data["label"]
    
    # Map condition names to label values
    condition_mapping = {v: k for k, v in WESAD_CONDITIONS.items()}
    target_labels = [condition_mapping[cond] for cond in conditions]
    
    # Create binary stress labels (1 for stress, 0 for non-stress)
    stress_label = condition_mapping["stress"]
    binary_labels = (labels == stress_label).astype(int)
    
    # Filter data for target conditions only
    condition_mask = np.isin(labels, target_labels)
    filtered_labels = binary_labels[condition_mask]
    
    # Extract and concatenate signal features
    features_list = []
    
    for signal_name in signals:
        if signal_name not in signal_data:
            logger.warning(f"Signal {signal_name} not found for {subject_id}")
            continue
        
        signal_values = signal_data[signal_name][condition_mask]
        
        # Handle multi-dimensional signals (e.g., ACC has 3 axes)
        if signal_values.ndim > 1:
            signal_values = signal_values.reshape(len(signal_values), -1)
        else:
            signal_values = signal_values.reshape(-1, 1)
        
        features_list.append(signal_values)
    
    if not features_list:
        raise WESADDatasetError(f"No valid signals found for {subject_id}")
    
    # Combine all signals
    combined_signals = np.hstack(features_list)
    
    # Extract features using sliding windows
    window_features, window_labels = _extract_windowed_features(
        signals=combined_signals,
        labels=filtered_labels,
        window_size=window_size,
        overlap=overlap,
        sampling_rate=WESAD_SIGNALS[sensor_location][signals[0]],  # Use first signal's rate
    )
    
    return window_features, window_labels


def _extract_windowed_features(
    signals: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    overlap: float,
    sampling_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract statistical features from sliding windows.
    
    Args:
        signals: Signal data (n_samples, n_signals).
        labels: Labels for each sample.
        window_size: Window size in seconds.
        overlap: Window overlap ratio (0.0-1.0).
        sampling_rate: Sampling rate in Hz.
    
    Returns:
        Tuple of (window_features, window_labels).
    """
    window_samples = window_size * sampling_rate
    step_samples = int(window_samples * (1 - overlap))
    
    if window_samples > len(signals):
        logger.warning(f"Window size ({window_samples}) larger than signal length ({len(signals)})")
        window_samples = len(signals)
    
    features = []
    window_labels = []
    
    for start_idx in range(0, len(signals) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        
        window_signals = signals[start_idx:end_idx]
        window_label_segment = labels[start_idx:end_idx]
        
        # Use majority vote for window label
        window_label = int(np.round(np.mean(window_label_segment)))
        
        # Extract statistical features for each signal channel
        window_features = []
        for channel in range(window_signals.shape[1]):
            channel_data = window_signals[:, channel]
            
            # Statistical features: mean, std, min, max, median
            features_channel = [
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data),
            ]
            window_features.extend(features_channel)
        
        features.append(window_features)
        window_labels.append(window_label)
    
    return np.array(features), np.array(window_labels)


def partition_wesad_by_subjects(
    data_dir: Optional[Union[str, Path]] = None,
    num_clients: int = 5,
    signals: List[str] = ["BVP", "EDA", "ACC", "TEMP"],
    sensor_location: str = "wrist",
    conditions: List[str] = ["baseline", "stress"],
    **kwargs
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Partition WESAD dataset by subjects for federated learning.

    Creates federated data partitions by assigning subjects to different
    clients, ensuring no subject appears in multiple clients (realistic
    federated scenario for healthcare data).

    Args:
        data_dir: Path to WESAD dataset directory.
        num_clients: Number of federated clients to create.
        signals: List of physiological signals to extract.
        sensor_location: Sensor location ('wrist' or 'chest').
        conditions: List of experimental conditions.
        **kwargs: Additional arguments passed to load_wesad_dataset.

    Returns:
        List of (X_train, X_test, y_train, y_test) tuples for each client.

    Raises:
        ValueError: If num_clients exceeds number of available subjects.

    Example:
        ```python
        # Partition WESAD into 3 federated clients
        client_datasets = partition_wesad_by_subjects(
            num_clients=3,
            signals=['BVP', 'EDA'],
            test_size=0.2
        )
        
        # Access data for first client
        X_train_0, X_test_0, y_train_0, y_test_0 = client_datasets[0]
        ```
    """
    if num_clients > len(WESAD_SUBJECTS):
        raise ValueError(
            f"Cannot create {num_clients} clients with only "
            f"{len(WESAD_SUBJECTS)} subjects"
        )
    
    # Shuffle subjects for random assignment
    np.random.seed(kwargs.get('random_state', 42))
    shuffled_subjects = np.random.permutation(WESAD_SUBJECTS)
    
    # Distribute subjects among clients
    subjects_per_client = len(WESAD_SUBJECTS) // num_clients
    extra_subjects = len(WESAD_SUBJECTS) % num_clients
    
    client_datasets = []
    subject_idx = 0
    
    for client_id in range(num_clients):
        # Calculate number of subjects for this client
        num_subjects_client = subjects_per_client
        if client_id < extra_subjects:
            num_subjects_client += 1
        
        # Get subjects for this client
        client_subjects = shuffled_subjects[
            subject_idx:subject_idx + num_subjects_client
        ].tolist()
        subject_idx += num_subjects_client
        
        # Load data for this client's subjects
        try:
            X_train, X_test, y_train, y_test = load_wesad_dataset(
                data_dir=data_dir,
                subjects=client_subjects,
                signals=signals,
                sensor_location=sensor_location,
                conditions=conditions,
                **kwargs
            )
            
            client_datasets.append((X_train, X_test, y_train, y_test))
            
            logger.info(f"Client {client_id}: subjects {client_subjects}, "
                       f"train: {len(X_train)}, test: {len(X_test)} windows")
                       
        except Exception as e:
            logger.error(f"Failed to create client {client_id} with subjects "
                        f"{client_subjects}: {e}")
            # Create empty dataset for failed client
            client_datasets.append((
                np.array([]).reshape(0, 0),
                np.array([]).reshape(0, 0),
                np.array([]),
                np.array([])
            ))
    
    return client_datasets
