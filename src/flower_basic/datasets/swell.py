"""SWELL-KW Dataset Loader for Stress Detection in Knowledge Work.

This module provides utilities for loading and preprocessing the SWELL-KW dataset,
which contains multimodal stress indicators from computer interaction, facial
expressions, body posture, and physiological signals.

Dataset Citation:
    Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., & Kraaij, W. (2014).
    The SWELL knowledge work dataset for stress and user modeling research.
    Proceedings of the 16th international conference on multimodal interaction.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CANONICAL_SCL_MAP = {
    "scl_mean": "eda_mean",
    "scl_std": "eda_std",
    "scl_min": "eda_min",
    "scl_max": "eda_max",
    "scl_median": "eda_median",
}


class SWELLDatasetError(Exception):
    """Exception raised for SWELL dataset loading errors."""

    pass


def load_swell_dataset(
    data_dir: Union[str, Path] = "data/SWELL",
    modalities: Optional[List[str]] = None,
    subjects: Optional[List[int]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize_features: bool = True,
    return_subject_info: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]],
]:
    """Load SWELL-KW dataset for stress detection in knowledge work."""

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SWELLDatasetError(f"SWELL dataset directory not found: {data_dir}")

    if modalities is None:
        modalities = ["computer", "facial", "posture", "physiology"]

    valid_modalities = {"computer", "facial", "posture", "physiology"}
    if not set(modalities).issubset(valid_modalities):
        invalid = set(modalities) - valid_modalities
        raise ValueError(
            f"Invalid modalities: {invalid}. Valid options: {valid_modalities}"
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0.0 and 1.0")

    if subjects is not None:
        normalized_subjects: Set[str] = set()
        for subject in subjects:
            try:
                value = int(subject)
            except (TypeError, ValueError) as exc:
                raise ValueError("Subject IDs must be between 1 and 25") from exc
            if not 1 <= value <= 25:
                raise ValueError("Subject IDs must be between 1 and 25")
            normalized_subjects.add(str(value))
    else:
        normalized_subjects = None

    feature_dir = data_dir / "3 - Feature dataset" / "per sensor"

    modality_files = {
        "computer": "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv",
        "facial": "B - Facial expressions features (FaceReaderAllData_final (NaN is 999))-sheet_1.csv",
        "posture": "C - Body posture features (Kinect C (position - per minute))- Sheet_1.csv",
        "physiology": "D - Physiology features (HR_HRV_SCL - final).csv",
    }

    dataframes: List[pd.DataFrame] = []
    feature_info: Dict[str, Dict[str, object]] = {}

    for modality in modalities:
        file_path = feature_dir / modality_files[modality]
        if not file_path.exists():
            raise SWELLDatasetError(f"Feature file not found: {file_path}")

        try:
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1")

            df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
            if "pp" in df.columns:
                df = df.rename(columns={"pp": "participant"})
            elif "subject" in df.columns and "participant" not in df.columns:
                df = df.rename(columns={"subject": "participant"})

            if modality == "facial":
                df = df.replace(999, np.nan)
            if modality == "physiology":
                df = df.rename(
                    columns={col: CANONICAL_SCL_MAP.get(col, col) for col in df.columns}
                )

            feature_info[modality] = {
                "n_features": len(df.columns)
                - len(
                    [
                        col
                        for col in ["participant", "subject", "condition"]
                        if col in df.columns
                    ]
                ),
                "missing_ratio": float(
                    df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
                ),
                "feature_names": [
                    col
                    for col in df.columns
                    if col not in {"participant", "subject", "condition"}
                ],
            }

            dataframes.append(df)

        except Exception as exc:  # pragma: no cover
            raise SWELLDatasetError(f"Error loading {modality} features: {exc}")

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merge_cols: List[str] = []
        for candidate in ["participant", "subject"]:
            if candidate in df.columns and candidate in merged_df.columns:
                merge_cols.append(candidate)
        if "condition" in df.columns and "condition" in merged_df.columns:
            merge_cols.append("condition")

        if not merge_cols:
            raise SWELLDatasetError("Could not determine merge keys for modalities")

        merged_df = pd.merge(merged_df, df, on=merge_cols, how="inner")

    subject_col = "participant" if "participant" in merged_df.columns else "subject"
    merged_df[subject_col] = merged_df[subject_col].astype(str)

    if normalized_subjects is not None:
        merged_df = merged_df[merged_df[subject_col].isin(normalized_subjects)]
        if merged_df.empty:
            raise SWELLDatasetError("No rows remaining after subject filtering")

    feature_columns = [
        col for col in merged_df.columns if col not in {subject_col, "condition"}
    ]

    features_df = merged_df[feature_columns].apply(pd.to_numeric, errors="coerce")

    conditions = (
        merged_df["condition"].astype(str).str.strip().str.lower().str.replace("_", " ")
    )
    condition_mapping = {
        "no stress": 0,
        "control": 0,
        "neutral": 0,
        "baseline": 0,
        "0": 0,
        "n": 0,
        "time pressure": 1,
        "interruption": 1,
        "interruptions": 1,
        "combined": 1,
        "stress": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "t": 1,
        "i": 1,
        "r": 1,
    }

    y = np.array(
        [condition_mapping.get(cond, 1) for cond in conditions], dtype=np.int64
    )
    if len(np.unique(y)) < 2:
        y = LabelEncoder().fit_transform(conditions)

    if features_df.isnull().to_numpy().any():
        warnings.warn(
            f"Found {int(features_df.isnull().sum().sum())} missing values. Filling with feature means.",
            UserWarning,
        )
        features_df = features_df.fillna(features_df.mean())
    features_df = features_df.fillna(0.0)

    X = features_df.to_numpy(dtype=np.float32, copy=False)

    feature_variances = np.var(X, axis=0)
    valid_features = feature_variances > 1e-8
    if not np.all(valid_features):
        warnings.warn(
            f"Removed {int((~valid_features).sum())} features with zero variance.",
            UserWarning,
        )
        X = X[:, valid_features]
        feature_columns = [
            col for idx, col in enumerate(feature_columns) if valid_features[idx]
        ]

    unique_subjects = merged_df[subject_col].unique()
    if len(unique_subjects) < 2:
        raise SWELLDatasetError("Not enough subjects to perform a split")

    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    subject_series = merged_df[subject_col].to_numpy()
    train_mask = np.isin(subject_series, train_subjects)
    test_mask = np.isin(subject_series, test_subjects)

    X_train_raw = X[train_mask]
    X_test_raw = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    train_subject_ids = subject_series[train_mask]
    test_subject_ids = subject_series[test_mask]

    if normalize_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    if return_subject_info:

        def _distribution(values: np.ndarray) -> Dict[int, int]:
            unique, counts = np.unique(values, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, counts)}

        info: Dict[str, object] = {
            "modalities": modalities,
            "feature_names": feature_columns,
            "subjects": unique_subjects.tolist(),
            "n_subjects": int(unique_subjects.size),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "train_subjects": train_subjects.tolist(),
            "test_subjects": test_subjects.tolist(),
            "train_subject_ids": train_subject_ids.tolist(),
            "test_subject_ids": test_subject_ids.tolist(),
            "class_distribution": _distribution(y),
            "train_class_distribution": _distribution(y_train),
            "test_class_distribution": _distribution(y_test),
            "feature_info": feature_info,
        }
        if normalize_features:
            info["scaler_mean"] = scaler.mean_.astype(float).tolist()
            info["scaler_scale"] = scaler.scale_.astype(float).tolist()

        return X_train, X_test, y_train, y_test, info

    return X_train, X_test, y_train, y_test


def partition_swell_by_subjects(
    data_dir: Union[str, Path] = "data/SWELL",
    n_partitions: int = 5,
    modalities: Optional[List[str]] = None,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Partition SWELL dataset by subjects for federated learning.

    Creates subject-based partitions where each partition contains data from
    a disjoint set of subjects, simulating realistic federated learning
    scenarios where different clients have data from different users.

    Args:
        data_dir: Path to SWELL dataset directory.
        n_partitions: Number of partitions to create (2-25).
        modalities: List of modalities to include (see load_swell_dataset).
        random_state: Random seed for reproducible partitioning.

    Returns:
        List of (X_train, X_test, y_train, y_test) tuples, one per partition.

    Raises:
        ValueError: If n_partitions is invalid or insufficient subjects.
        SWELLDatasetError: If dataset loading fails.

    Example:
        >>> # Create 5 federated partitions
        >>> partitions = partition_swell_by_subjects(n_partitions=5)
        >>> for i, (X_train, X_test, y_train, y_test) in enumerate(partitions):
        ...     print(f"Partition {i}: {X_train.shape[0]} training samples")
    """
    if not 2 <= n_partitions <= 25:
        raise ValueError(
            "n_partitions must be between 2 and 25 (number of SWELL subjects)"
        )

    data_dir = Path(data_dir)

    # Get all available subjects
    all_subjects = list(range(1, 26))  # SWELL has subjects 1-25

    # Verify we have enough subjects
    if n_partitions > len(all_subjects):
        raise ValueError(
            f"Cannot create {n_partitions} partitions with only {len(all_subjects)} subjects"
        )

    # Shuffle subjects for random assignment
    np.random.seed(random_state)
    shuffled_subjects = np.random.permutation(all_subjects)

    # Create subject partitions
    subjects_per_partition = len(all_subjects) // n_partitions
    partitions = []

    for i in range(n_partitions):
        start_idx = i * subjects_per_partition

        # Last partition gets remaining subjects
        if i == n_partitions - 1:
            partition_subjects = shuffled_subjects[start_idx:].tolist()
        else:
            end_idx = start_idx + subjects_per_partition
            partition_subjects = shuffled_subjects[start_idx:end_idx].tolist()

        try:
            # Load data for this partition's subjects
            X_train, X_test, y_train, y_test = load_swell_dataset(
                data_dir=data_dir,
                modalities=modalities,
                subjects=partition_subjects,
                random_state=random_state,
            )

            partitions.append((X_train, X_test, y_train, y_test))

        except Exception as e:
            raise SWELLDatasetError(
                f"Error creating partition {i} with subjects {partition_subjects}: {e}"
            )

    return partitions


def get_swell_info(data_dir: Union[str, Path] = "data/SWELL") -> Dict:
    """Get comprehensive information about the SWELL dataset.

    Args:
        data_dir: Path to SWELL dataset directory.

    Returns:
        Dictionary with dataset information including modalities, subjects,
        feature counts, and data quality metrics.
    """
    try:
        _, _, _, _, info = load_swell_dataset(
            data_dir=data_dir, return_subject_info=True
        )

        info["description"] = (
            "SWELL-KW: Multimodal dataset for stress detection in knowledge work. "
            "Contains computer interaction, facial expressions, body posture, "
            "and physiological signals from 25 participants under different stress conditions."
        )

        info["citation"] = (
            "Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., & Kraaij, W. (2014). "
            "The SWELL knowledge work dataset for stress and user modeling research. "
            "Proceedings of the 16th international conference on multimodal interaction."
        )

        return info

    except Exception as e:
        return {"error": str(e), "status": "Dataset not available"}


# Compatibility aliases for common usage patterns
load_swell = load_swell_dataset
partition_swell = partition_swell_by_subjects
