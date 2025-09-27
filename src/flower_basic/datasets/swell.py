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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """Load SWELL-KW dataset for stress detection in knowledge work.
    
    The SWELL dataset contains multimodal stress indicators from 25 participants
    performing knowledge work tasks under different stress conditions:
    - No stress (control)
    - Time pressure
    - Interruptions
    - Combined stress (time pressure + interruptions)
    
    Args:
        data_dir: Path to SWELL dataset directory containing feature files.
        modalities: List of modalities to include. Options:
            - 'computer': Computer interaction features (mouse, keyboard, apps)
            - 'facial': Facial expression features (emotions, head pose)
            - 'posture': Body posture features (Kinect skeleton data)
            - 'physiology': Physiological features (HR, HRV, SCL)
            If None, includes all available modalities.
        subjects: List of subject IDs to include (1-25). If None, includes all.
        test_size: Fraction of data to use for testing (0.0-1.0).
        random_state: Random seed for reproducible train/test split.
        normalize_features: Whether to standardize features (zero mean, unit variance).
        return_subject_info: Whether to return subject metadata dictionary.
    
    Returns:
        If return_subject_info is False:
            X_train: Training features array of shape (n_train_samples, n_features)
            X_test: Test features array of shape (n_test_samples, n_features)
            y_train: Training labels array of shape (n_train_samples,)
            y_test: Test labels array of shape (n_test_samples,)
        
        If return_subject_info is True:
            Same as above plus:
            subject_info: Dictionary containing subject metadata and feature info
    
    Raises:
        SWELLDatasetError: If dataset files are not found or corrupted.
        ValueError: If invalid parameters are provided.
    
    Example:
        >>> # Load all modalities for binary stress classification
        >>> X_train, X_test, y_train, y_test = load_swell_dataset()
        >>> print(f"Training data: {X_train.shape}, Labels: {np.unique(y_train)}")
        
        >>> # Load only physiological and computer interaction data
        >>> X_train, X_test, y_train, y_test = load_swell_dataset(
        ...     modalities=['physiology', 'computer']
        ... )
        
        >>> # Load specific subjects with metadata
        >>> X_train, X_test, y_train, y_test, info = load_swell_dataset(
        ...     subjects=[1, 2, 3], return_subject_info=True
        ... )
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise SWELLDatasetError(f"SWELL dataset directory not found: {data_dir}")
    
    # Validate parameters
    if modalities is None:
        modalities = ['computer', 'facial', 'posture', 'physiology']
    
    valid_modalities = {'computer', 'facial', 'posture', 'physiology'}
    if not set(modalities).issubset(valid_modalities):
        invalid = set(modalities) - valid_modalities
        raise ValueError(f"Invalid modalities: {invalid}. Valid options: {valid_modalities}")
    
    if subjects is not None:
        if not all(1 <= s <= 25 for s in subjects):
            raise ValueError("Subject IDs must be between 1 and 25")
    
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0.0 and 1.0")
    
    # Load feature files
    feature_dir = data_dir / "3 - Feature dataset" / "per sensor"
    
    modality_files = {
        'computer': "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv",
        'facial': "B - Facial expressions features (FaceReaderAllData_final (NaN is 999))-sheet_1.csv", 
        'posture': "C - Body posture features (Kinect - final (annotated and selected))-sheet_1.csv",
        'physiology': "D - Physiology features (HR_HRV_SCL - final).csv"
    }
    
    dataframes = []
    feature_info = {}
    
    for modality in modalities:
        file_path = feature_dir / modality_files[modality]
        
        if not file_path.exists():
            raise SWELLDatasetError(f"Feature file not found: {file_path}")
        
        try:
            # Read CSV with robust encoding detection
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            
            # Handle missing values (NaN represented as 999 in some files)
            if modality == 'facial':
                df = df.replace(999, np.nan)
            
            # Store feature information
            feature_info[modality] = {
                'n_features': len(df.columns) - 2,  # Exclude subject and condition columns
                'feature_names': [col for col in df.columns if col not in ['subject', 'condition', 'participant']],
                'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            }
            
            dataframes.append(df)
            
        except Exception as e:
            raise SWELLDatasetError(f"Error loading {modality} features: {e}")
    
    # Merge dataframes on subject and condition
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        # Find common columns for merging
        merge_cols = ['subject'] if 'subject' in df.columns else ['participant']
        if 'condition' in df.columns:
            merge_cols.append('condition')
        
        merged_df = pd.merge(merged_df, df, on=merge_cols, how='inner')
    
    # Filter subjects if specified
    if subjects is not None:
        subject_col = 'subject' if 'subject' in merged_df.columns else 'participant'
        merged_df = merged_df[merged_df[subject_col].isin(subjects)]
    
    # Prepare features and labels
    feature_columns = [col for col in merged_df.columns 
                      if col not in ['subject', 'participant', 'condition']]
    
    X = merged_df[feature_columns].values
    
    # Handle condition labels - map to binary stress classification
    condition_mapping = {
        'No stress': 0,
        'Control': 0,
        'Neutral': 0,
        'Time pressure': 1,
        'Interruption': 1,
        'Interruptions': 1,
        'Combined': 1,  # Time pressure + interruptions
        'Stress': 1
    }
    
    conditions = merged_df['condition'].astype(str)
    label_encoder = LabelEncoder()
    
    # Try to map known conditions, otherwise use label encoder
    try:
        y = np.array([condition_mapping.get(cond, 1) for cond in conditions])
        if len(np.unique(y)) < 2:
            # Fallback to automatic encoding if mapping failed
            y = label_encoder.fit_transform(conditions)
    except:
        y = label_encoder.fit_transform(conditions)
    
    # Handle missing values
    if np.any(np.isnan(X)):
        warnings.warn(
            f"Found {np.sum(np.isnan(X))} missing values. Filling with feature means.",
            UserWarning
        )
        # Fill missing values with feature means
        for col_idx in range(X.shape[1]):
            col_mean = np.nanmean(X[:, col_idx])
            X[np.isnan(X[:, col_idx]), col_idx] = col_mean
    
    # Remove features with zero variance
    feature_variances = np.var(X, axis=0)
    valid_features = feature_variances > 1e-8
    
    if not np.all(valid_features):
        n_removed = np.sum(~valid_features)
        warnings.warn(
            f"Removed {n_removed} features with zero variance.",
            UserWarning
        )
        X = X[:, valid_features]
        feature_columns = [col for i, col in enumerate(feature_columns) if valid_features[i]]
    
    # Normalize features if requested
    if normalize_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if not return_subject_info:
        return X_train, X_test, y_train, y_test
    
    # Prepare subject information
    subject_info = {
        'n_subjects': len(merged_df['subject' if 'subject' in merged_df.columns else 'participant'].unique()),
        'n_samples': len(merged_df),
        'n_features': X.shape[1],
        'feature_names': feature_columns,
        'modalities': modalities,
        'feature_info': feature_info,
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'conditions_mapped': condition_mapping
    }
    
    return X_train, X_test, y_train, y_test, subject_info


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
        raise ValueError("n_partitions must be between 2 and 25 (number of SWELL subjects)")
    
    data_dir = Path(data_dir)
    
    # Get all available subjects
    all_subjects = list(range(1, 26))  # SWELL has subjects 1-25
    
    # Verify we have enough subjects
    if n_partitions > len(all_subjects):
        raise ValueError(f"Cannot create {n_partitions} partitions with only {len(all_subjects)} subjects")
    
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
            raise SWELLDatasetError(f"Error creating partition {i} with subjects {partition_subjects}: {e}")
    
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
            data_dir=data_dir,
            return_subject_info=True
        )
        
        info['description'] = (
            "SWELL-KW: Multimodal dataset for stress detection in knowledge work. "
            "Contains computer interaction, facial expressions, body posture, "
            "and physiological signals from 25 participants under different stress conditions."
        )
        
        info['citation'] = (
            "Koldijk, S., Sappelli, M., Verberne, S., Neerincx, M. A., & Kraaij, W. (2014). "
            "The SWELL knowledge work dataset for stress and user modeling research. "
            "Proceedings of the 16th international conference on multimodal interaction."
        )
        
        return info
        
    except Exception as e:
        return {'error': str(e), 'status': 'Dataset not available'}


# Compatibility aliases for common usage patterns
load_swell = load_swell_dataset
partition_swell = partition_swell_by_subjects
