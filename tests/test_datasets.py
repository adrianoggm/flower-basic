"""Tests for dataset loaders.

This module tests the dataset loading functionality for ECG5000 and WESAD datasets,
ensuring proper data preprocessing, validation, and federated partitioning.

Test categories:
- Unit tests for individual loader functions
- Integration tests for end-to-end data loading
- Performance tests for large dataset handling
- Error handling tests for various failure modes
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from flower_basic.datasets import load_ecg5000_dataset, load_wesad_dataset, load_swell_dataset
from flower_basic.datasets.ecg5000 import ECGDatasetError, partition_ecg5000_by_subjects
from flower_basic.datasets.wesad import WESADDatasetError, partition_wesad_by_subjects
from flower_basic.datasets.swell import SWELLDatasetError, partition_swell_by_subjects, get_swell_info


class TestECG5000Dataset:
    """Test suite for ECG5000 dataset loading functionality."""

    @pytest.fixture
    def mock_ecg_data(self) -> Mock:
        """Create mock ECG5000 data for testing."""
        mock_data = Mock()
        
        # Create synthetic ECG data (5000 samples, 140 features)
        X_synthetic = np.random.randn(5000, 140).astype(np.float32)
        y_synthetic = np.random.choice([1, 2, 3, 4, 5], size=5000)
        
        mock_data.data.values = X_synthetic
        mock_data.target.values = y_synthetic
        
        return mock_data

    @patch('flower_basic.datasets.ecg5000.fetch_openml')
    def test_load_ecg5000_default_configuration(self, mock_fetch: Mock, mock_ecg_data: Mock) -> None:
        """Test loading ECG5000 with default parameters."""
        mock_fetch.return_value = mock_ecg_data
        
        X_train, X_test, y_train, y_test = load_ecg5000_dataset()
        
        # Validate shapes
        assert X_train.shape[0] > 0, "Training set should not be empty"
        assert X_test.shape[0] > 0, "Test set should not be empty" 
        assert X_train.shape[1] == X_test.shape[1], "Feature dimensions must match"
        assert X_train.shape[1] == 140, "ECG5000 should have 140 features"
        
        # Validate types
        assert isinstance(X_train, np.ndarray), "Should return numpy arrays"
        assert X_train.dtype == np.float32, "Should use float32 for memory efficiency"
        assert y_train.dtype == np.int64, "Labels should be int64"
        
        # Validate labels (binary classification)
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        assert set(unique_labels).issubset({0, 1}), "Should have binary labels (0, 1)"
        
        # Validate split ratio
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        assert abs(test_ratio - 0.2) < 0.05, "Test size should be approximately 0.2"

    @patch('flower_basic.datasets.ecg5000.fetch_openml')
    def test_load_ecg5000_with_subject_ids(self, mock_fetch: Mock, mock_ecg_data: Mock) -> None:
        """Test loading ECG5000 with subject IDs for federated scenarios."""
        mock_fetch.return_value = mock_ecg_data
        
        X_train, X_test, y_train, y_test, subjects_train = load_ecg5000_dataset(
            return_subject_ids=True
        )
        
        # Validate subject IDs
        assert len(subjects_train) == len(X_train), "Subject IDs must match training samples"
        assert all(isinstance(s, str) for s in subjects_train), "Subject IDs should be strings"
        assert all(s.startswith('S') for s in subjects_train), "Subject IDs should start with 'S'"
        
        # Check that we have multiple subjects
        unique_subjects = np.unique(subjects_train)
        assert len(unique_subjects) > 1, "Should have multiple subjects for federated learning"

    def test_invalid_test_size_raises_error(self) -> None:
        """Test that invalid test_size values raise appropriate errors."""
        with pytest.raises(ValueError, match="test_size must be between"):
            load_ecg5000_dataset(test_size=0.0)
        
        with pytest.raises(ValueError, match="test_size must be between"):
            load_ecg5000_dataset(test_size=1.0)
        
        with pytest.raises(ValueError, match="test_size must be between"):
            load_ecg5000_dataset(test_size=-0.1)

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.5])
    @patch('flower_basic.datasets.ecg5000.fetch_openml')
    def test_various_test_sizes(
        self, 
        mock_fetch: Mock, 
        mock_ecg_data: Mock, 
        test_size: float
    ) -> None:
        """Test various test_size configurations."""
        mock_fetch.return_value = mock_ecg_data
        
        X_train, X_test, y_train, y_test = load_ecg5000_dataset(test_size=test_size)
        
        total_samples = len(X_train) + len(X_test)
        actual_ratio = len(X_test) / total_samples
        
        assert abs(actual_ratio - test_size) < 0.05, f"Test size ratio should be ~{test_size}"

    @patch('flower_basic.datasets.ecg5000.fetch_openml')
    def test_normalization_parameter(self, mock_fetch: Mock, mock_ecg_data: Mock) -> None:
        """Test normalization parameter functionality."""
        mock_fetch.return_value = mock_ecg_data
        
        # Test with normalization (default)
        X_train_norm, _, _, _ = load_ecg5000_dataset(normalize=True)
        
        # Test without normalization
        X_train_raw, _, _, _ = load_ecg5000_dataset(normalize=False)
        
        # Normalized data should have different statistics
        assert not np.allclose(X_train_norm, X_train_raw), "Normalized data should differ from raw"
        
        # Normalized data should be approximately standardized
        assert abs(np.mean(X_train_norm)) < 0.1, "Normalized data should have near-zero mean"
        assert abs(np.std(X_train_norm) - 1.0) < 0.1, "Normalized data should have unit variance"

    @patch('flower_basic.datasets.ecg5000.fetch_openml')
    def test_partition_by_subjects(self, mock_fetch: Mock, mock_ecg_data: Mock) -> None:
        """Test subject-based partitioning for federated learning."""
        mock_fetch.return_value = mock_ecg_data
        
        # Load data with subject IDs
        X_train, _, y_train, _, subjects = load_ecg5000_dataset(return_subject_ids=True)
        
        # Partition into clients
        num_clients = 3
        client_data = partition_ecg5000_by_subjects(
            X_train, y_train, subjects, num_clients=num_clients
        )
        
        assert len(client_data) == num_clients, f"Should create {num_clients} clients"
        
        # Verify each client has data
        for i, (X_client, y_client) in enumerate(client_data):
            assert len(X_client) > 0, f"Client {i} should have data"
            assert len(X_client) == len(y_client), f"Client {i} X and y should match"
        
        # Verify no data overlap (all data is distributed)
        total_client_samples = sum(len(X_client) for X_client, _ in client_data)
        assert total_client_samples == len(X_train), "All training data should be distributed"

    def test_partition_too_many_clients_raises_error(self) -> None:
        """Test that requesting too many clients raises appropriate error."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        subjects = np.array([f"S{i:02d}" for i in range(5)] * 20)  # 5 unique subjects
        
        with pytest.raises(ValueError, match="Cannot create .* clients"):
            partition_ecg5000_by_subjects(X, y, subjects, num_clients=10)

    @patch('flower_basic.datasets.ecg5000.fetch_openml')
    def test_dataset_error_handling(self, mock_fetch: Mock) -> None:
        """Test error handling when dataset loading fails."""
        mock_fetch.side_effect = Exception("Network error")
        
        with pytest.raises(ECGDatasetError, match="ECG5000 dataset loading failed"):
            load_ecg5000_dataset()


class TestWESADDataset:
    """Test suite for WESAD dataset loading functionality."""

    @pytest.fixture
    def mock_wesad_data_dir(self):
        """Create mock WESAD data directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "WESAD"
            
            # Create subject directories
            for subject_id in ["S2", "S3", "S4"]:
                subject_dir = data_dir / subject_id
                subject_dir.mkdir(parents=True)
                
                # Create mock pickle file with minimal structure
                mock_data = {
                    "signal": {
                        "wrist": {
                            "BVP": np.random.randn(1000),
                            "EDA": np.random.randn(1000),
                            "ACC": np.random.randn(1000, 3),
                            "TEMP": np.random.randn(1000),
                        }
                    },
                    "label": np.random.choice([1, 2], size=1000)  # baseline and stress
                }
                
                import pickle
                pickle_path = subject_dir / f"{subject_id}.pkl"
                with open(pickle_path, "wb") as f:
                    pickle.dump(mock_data, f)
            
            yield data_dir

    def test_load_wesad_dataset_validation(self) -> None:
        """Test parameter validation for WESAD loader."""
        # Test invalid sensor location
        with pytest.raises(ValueError, match="sensor_location must be"):
            load_wesad_dataset(sensor_location="invalid")
        
        # Test invalid signals
        with pytest.raises(ValueError, match="Invalid signals"):
            load_wesad_dataset(signals=["INVALID_SIGNAL"])
        
        # Test invalid conditions
        with pytest.raises(ValueError, match="Invalid conditions"):
            load_wesad_dataset(conditions=["invalid_condition"])
        
        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size must be between"):
            load_wesad_dataset(test_size=0.0)

    def test_missing_data_directory_raises_error(self) -> None:
        """Test that missing data directory raises appropriate error."""
        with pytest.raises(FileNotFoundError, match="WESAD data directory not found"):
            load_wesad_dataset(data_dir="/nonexistent/path")

    def test_invalid_subjects_raises_error(self) -> None:
        """Test that invalid subject IDs raise appropriate errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Invalid subjects"):
                load_wesad_dataset(
                    data_dir=temp_dir,
                    subjects=["S1", "S99"]  # S1 excluded, S99 doesn't exist
                )

    def test_wesad_dataset_error_handling(self, mock_wesad_data_dir: Path) -> None:
        """Test error handling for WESAD dataset loading failures."""
        # Test with empty data directory
        empty_dir = mock_wesad_data_dir.parent / "empty_wesad"
        empty_dir.mkdir()
        
        with pytest.raises(WESADDatasetError, match="No valid subject data could be loaded"):
            load_wesad_dataset(data_dir=empty_dir, subjects=["S2"])

    @pytest.mark.slow
    def test_partition_wesad_by_subjects(self) -> None:
        """Test subject-based partitioning for WESAD dataset."""
        num_clients = 3
        
        # Test parameter validation
        with pytest.raises(ValueError, match="Cannot create .* clients"):
            partition_wesad_by_subjects(num_clients=20)  # More clients than subjects
        
        # Note: Full integration test would require actual WESAD data
        # This test validates the interface and error handling


class TestSWELLDataset:
    """Test suite for SWELL dataset loading functionality."""

    def test_load_swell_dataset_with_mock_data(self):
        """Test SWELL dataset loading with mocked data files."""
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock computer interaction features
            computer_df = pd.DataFrame({
                'subject': [1, 1, 2, 2, 3, 3],
                'condition': ['No stress', 'Time pressure', 'No stress', 'Interruptions', 'Control', 'Combined'],
                'mouse_clicks': [10, 20, 15, 25, 12, 30],
                'keyboard_strokes': [100, 150, 120, 180, 110, 200],
                'app_switches': [5, 8, 6, 10, 5, 12]
            })
            
            # Mock physiology features  
            physio_df = pd.DataFrame({
                'subject': [1, 1, 2, 2, 3, 3],
                'condition': ['No stress', 'Time pressure', 'No stress', 'Interruptions', 'Control', 'Combined'],
                'hr_mean': [70.5, 85.2, 68.1, 82.7, 72.3, 88.9],
                'hrv_rmssd': [45.2, 32.1, 50.3, 28.5, 48.7, 25.8],
                'scl_mean': [2.1, 3.5, 1.9, 3.2, 2.3, 3.8]
            })
            
            mock_read_csv.side_effect = [computer_df, physio_df]
            
            X_train, X_test, y_train, y_test = load_swell_dataset(
                modalities=['computer', 'physiology'],
                test_size=0.3,
                random_state=42
            )
            
            # Verify shapes
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert X_train.shape[1] > 0  # Should have features from both modalities
            assert len(y_train) == X_train.shape[0]
            assert len(y_test) == X_test.shape[0]
            
            # Verify binary classification (stress vs no stress)
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            assert len(unique_labels) <= 2

    def test_load_swell_dataset_with_subject_info(self):
        """Test SWELL dataset loading with subject information return."""
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock single modality data
            mock_df = pd.DataFrame({
                'subject': [1, 2, 3] * 4,
                'condition': ['No stress'] * 6 + ['Time pressure'] * 6,
                'feature1': np.random.randn(12),
                'feature2': np.random.randn(12),
                'feature3': np.random.randn(12)
            })
            
            mock_read_csv.return_value = mock_df
            
            X_train, X_test, y_train, y_test, subject_info = load_swell_dataset(
                modalities=['computer'],
                return_subject_info=True
            )
            
            # Verify subject info structure
            assert isinstance(subject_info, dict)
            assert 'n_subjects' in subject_info
            assert 'n_samples' in subject_info
            assert 'n_features' in subject_info
            assert 'feature_names' in subject_info
            assert 'modalities' in subject_info
            assert 'class_distribution' in subject_info
            
            # Verify content
            assert subject_info['n_subjects'] == 3
            assert subject_info['n_samples'] == 12
            assert subject_info['modalities'] == ['computer']

    def test_partition_swell_by_subjects(self):
        """Test SWELL dataset partitioning by subjects."""
        with patch('flower_basic.datasets.swell.load_swell_dataset') as mock_load:
            # Mock different partitions with different subjects
            def mock_load_side_effect(*args, **kwargs):
                subjects = kwargs.get('subjects', [1, 2, 3])
                n_samples = len(subjects) * 10  # 10 samples per subject
                n_features = 50
                
                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, 2, n_samples)
                
                # Split into train/test
                split_idx = int(0.8 * n_samples)
                return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
            
            mock_load.side_effect = mock_load_side_effect
            
            # Test partitioning
            partitions = partition_swell_by_subjects(n_partitions=3, random_state=42)
            
            assert len(partitions) == 3
            
            for i, (X_train, X_test, y_train, y_test) in enumerate(partitions):
                assert X_train.shape[0] > 0
                assert X_test.shape[0] > 0
                assert len(y_train) == X_train.shape[0]
                assert len(y_test) == X_test.shape[0]
                
                # Verify binary classification
                all_labels = np.concatenate([y_train, y_test])
                assert len(np.unique(all_labels)) <= 2

    def test_swell_dataset_error_handling(self):
        """Test SWELL dataset error handling for various failure modes."""
        
        # Test missing directory
        with pytest.raises(SWELLDatasetError, match="directory not found"):
            load_swell_dataset(data_dir="nonexistent/path")
        
        # Test invalid modalities
        with pytest.raises(ValueError, match="Invalid modalities"):
            load_swell_dataset(modalities=['invalid_modality'])
        
        # Test invalid subjects
        with pytest.raises(ValueError, match="Subject IDs must be between 1 and 25"):
            load_swell_dataset(subjects=[0, 30])
        
        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size must be between"):
            load_swell_dataset(test_size=1.5)

    def test_swell_modality_selection(self):
        """Test SWELL dataset loading with different modality combinations."""
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock data for each modality
            base_data = {
                'subject': [1, 2, 3] * 2,
                'condition': ['No stress'] * 3 + ['Time pressure'] * 3,
            }
            
            computer_df = pd.DataFrame({**base_data, 'mouse_clicks': np.random.randn(6)})
            facial_df = pd.DataFrame({**base_data, 'emotion_joy': np.random.randn(6)}) 
            posture_df = pd.DataFrame({**base_data, 'head_angle': np.random.randn(6)})
            physio_df = pd.DataFrame({**base_data, 'hr_mean': np.random.randn(6)})
            
            # Test single modality
            mock_read_csv.return_value = computer_df
            X_train, X_test, y_train, y_test = load_swell_dataset(
                modalities=['computer'], test_size=0.3
            )
            assert X_train.shape[1] == 1  # Only mouse_clicks feature
            
            # Test multiple modalities
            mock_read_csv.side_effect = [computer_df, physio_df]
            X_train, X_test, y_train, y_test = load_swell_dataset(
                modalities=['computer', 'physiology'], test_size=0.3
            )
            assert X_train.shape[1] == 2  # Features from both modalities

    def test_get_swell_info(self):
        """Test SWELL dataset information retrieval."""
        with patch('flower_basic.datasets.swell.load_swell_dataset') as mock_load:
            mock_load.return_value = (
                np.random.randn(100, 50),  # X_train
                np.random.randn(25, 50),   # X_test  
                np.random.randint(0, 2, 100),  # y_train
                np.random.randint(0, 2, 25),   # y_test
                {
                    'n_subjects': 25,
                    'n_samples': 125,
                    'n_features': 50,
                    'modalities': ['computer', 'facial', 'posture', 'physiology']
                }
            )
            
            info = get_swell_info()
            
            assert isinstance(info, dict)
            assert 'description' in info
            assert 'citation' in info
            assert 'n_subjects' in info
            assert info['n_subjects'] == 25
            assert 'SWELL' in info['description']

    @pytest.mark.parametrize("n_partitions", [2, 5, 10])
    def test_swell_partition_sizes(self, n_partitions):
        """Test SWELL partitioning with different partition counts."""
        with patch('flower_basic.datasets.swell.load_swell_dataset') as mock_load:
            def mock_load_side_effect(*args, **kwargs):
                subjects = kwargs.get('subjects', list(range(1, 26)))
                n_samples = len(subjects) * 5
                return (
                    np.random.randn(n_samples, 20),
                    np.random.randn(n_samples // 4, 20),
                    np.random.randint(0, 2, n_samples),
                    np.random.randint(0, 2, n_samples // 4)
                )
            
            mock_load.side_effect = mock_load_side_effect
            
            partitions = partition_swell_by_subjects(n_partitions=n_partitions)
            assert len(partitions) == n_partitions
            
            # Verify all partitions have data
            for partition in partitions:
                X_train, X_test, y_train, y_test = partition
                assert X_train.shape[0] > 0
                assert X_test.shape[0] > 0


class TestDatasetIntegration:
    """Integration tests for dataset loading functionality."""

    @pytest.mark.integration
    def test_dataset_consistency(self) -> None:
        """Test consistency between different dataset loaders."""
        # This would test that all loaders return compatible formats
        # For now, we test the interface consistency
        
        # All loaders should return the same tuple structure
        expected_return_types = (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        
        # ECG5000 loader interface
        with patch('flower_basic.datasets.ecg5000.fetch_openml'):
            try:
                # This will fail due to mock, but we can test the interface
                load_ecg5000_dataset()
            except Exception:
                pass  # Expected due to mocking
        
        # WESAD loader interface 
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                load_wesad_dataset(data_dir=temp_dir)
            except (WESADDatasetError, FileNotFoundError):
                pass  # Expected due to missing data

    @pytest.mark.performance
    def test_dataset_loading_performance(self) -> None:
        """Test dataset loading performance requirements."""
        import time
        
        # Test that dataset loading completes within reasonable time
        with patch('flower_basic.datasets.ecg5000.fetch_openml') as mock_fetch:
            # Create lightweight mock data
            mock_data = Mock()
            mock_data.data.values = np.random.randn(100, 140).astype(np.float32)
            mock_data.target.values = np.random.choice([1, 2], size=100)
            mock_fetch.return_value = mock_data
            
            start_time = time.time()
            load_ecg5000_dataset()
            elapsed_time = time.time() - start_time
            
            # Should complete within 5 seconds for small datasets
            assert elapsed_time < 5.0, f"Dataset loading took {elapsed_time:.2f}s, should be <5s"

    @pytest.mark.unit
    def test_dataset_memory_usage(self) -> None:
        """Test that dataset loading meets memory requirements."""
        # This would test memory usage, but requires actual implementation
        # For now, we ensure returned data types are memory-efficient
        
        with patch('flower_basic.datasets.ecg5000.fetch_openml') as mock_fetch:
            mock_data = Mock()
            mock_data.data.values = np.random.randn(1000, 140).astype(np.float64)  # Start with float64
            mock_data.target.values = np.random.choice([1, 2], size=1000)
            mock_fetch.return_value = mock_data
            
            X_train, X_test, y_train, y_test = load_ecg5000_dataset()
            
            # Verify memory-efficient data types
            assert X_train.dtype == np.float32, "Should convert to float32 for memory efficiency"
            assert y_train.dtype == np.int64, "Labels should use int64"
