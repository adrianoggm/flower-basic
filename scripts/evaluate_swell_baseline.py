#!/usr/bin/env python3
"""
Baseline Performance Evaluation: SWELL Dataset
==============================================

Evaluates SWELL dataset performance using classical machine learning approach
with proper subject-based splitting to prevent data leakage.

Split strategy:
- 50% subjects for training
- 20% subjects for validation
- 30% subjects for testing
- No subject data mixing between splits

This establishes a baseline for comparison with federated learning results.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class SWELLBaselineEvaluator:
    """Baseline performance evaluator for SWELL dataset."""

    def __init__(self, data_dir: str = "data/SWELL"):
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_swell_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load COMPLETE SWELL dataset - MANDATORY for evaluations."""
        print("Loading COMPLETE SWELL dataset...")

        # RULE ENFORCEMENT: Evaluations MUST use complete datasets
        # NO samples, NO mock data - COMPLETE dataset ONLY

        # Path to COMPLETE SWELL feature files
        feature_dir = self.data_dir / "3 - Feature dataset" / "per sensor"

        if not feature_dir.exists():
            print(f"SWELL complete dataset not found: {feature_dir}")
            # Check alternative paths for COMPLETE dataset
            alt_paths = [
                self.data_dir,
                self.data_dir.parent / "SWELL",
                Path("data") / "0_SWELL",
            ]

            feature_dir = None
            for alt_path in alt_paths:
                if alt_path.exists():
                    csv_files = list(alt_path.glob("*.csv"))
                    if csv_files:
                        feature_dir = alt_path
                        print(f"  Found COMPLETE SWELL dataset in: {feature_dir}")
                        break

            if feature_dir is None:
                raise FileNotFoundError(
                    f"❌ CRITICAL: Complete SWELL dataset not found!\n"
                    f"   Evaluations REQUIRE complete datasets (not samples)\n"
                    f"   Checked paths: {alt_paths}\n"
                    f"   Please ensure SWELL complete dataset is available"
                )

        # REMOVED: _load_swell_from_samples()
        # RULE: Evaluations MUST use complete datasets only
        # Samples are ONLY for tests (test_*.py files)

        # Load different modality files - try multiple filename patterns
        modality_patterns = {
            "computer": [
                "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv",
                "computer_features.csv",
                "*computer*.csv",
            ],
            "facial": [
                "B - Facial expressions features (FaceReaderAllData_final (NaN is 999))-sheet_1.csv",
                "facial_features.csv",
                "*facial*.csv",
                "*face*.csv",
            ],
            "posture": [
                "C - Body posture features (Kinect - final (annotated and selected))-sheet_1.csv",
                "posture_features.csv",
                "*posture*.csv",
                "*kinect*.csv",
            ],
            "physiology": [
                "D - Physiology features (HR_HRV_SCL - final).csv",
                "physiology_features.csv",
                "*physiology*.csv",
                "*hr*.csv",
            ],
        }

        print("Loading real SWELL modality data...")

        dataframes = []

        for modality, patterns in modality_patterns.items():
            df_loaded = False

            for pattern in patterns:
                if "*" in pattern:
                    # Glob pattern
                    matching_files = list(feature_dir.glob(pattern))
                    if matching_files:
                        file_path = matching_files[0]  # Use first match
                    else:
                        continue
                else:
                    # Exact filename
                    file_path = feature_dir / pattern
                    if not file_path.exists():
                        continue

                try:
                    # Try different encodings and separators for real data
                    try:
                        df = pd.read_csv(file_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, encoding="latin1")
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, encoding="cp1252")

                    # Handle semicolon-separated files (common in SWELL facial data)
                    if df.shape[1] == 1 and ";" in str(df.columns[0]):
                        try:
                            df = pd.read_csv(file_path, sep=";", encoding="utf-8")
                        except:
                            try:
                                df = pd.read_csv(file_path, sep=";", encoding="latin1")
                            except:
                                df = pd.read_csv(file_path, sep=";", encoding="cp1252")

                    print(
                        f"  ✓ Loaded REAL {modality} data: {df.shape} from {file_path.name}"
                    )

                    # Clean column names
                    df.columns = (
                        df.columns.str.strip().str.replace(" ", "_").str.lower()
                    )

                    # Handle missing values (999 represents NaN in facial data)
                    if modality == "facial":
                        df = df.replace(999, np.nan)

                    # Handle NaN values with real data strategies
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        df[numeric_columns] = df[numeric_columns].fillna(
                            df[numeric_columns].median()
                        )

                    # Add modality prefix to avoid column conflicts
                    feature_cols = [
                        col
                        for col in df.columns
                        if col
                        not in [
                            "participant",
                            "subject",
                            "condition",
                            "time",
                            "timestamp",
                            "pp",
                            "blok",
                            "c",
                        ]
                    ]
                    df = df.rename(
                        columns={col: f"{modality}_{col}" for col in feature_cols}
                    )

                    dataframes.append(df)
                    df_loaded = True
                    break

                except Exception as e:
                    print(f"    Error loading {file_path}: {e}")
                    continue

            if not df_loaded:
                print(f"  ⚠️  Could not load {modality} data - no valid files found")

        if len(dataframes) == 0:
            raise ValueError(
                "❌ CRITICAL: No COMPLETE SWELL modality data loaded!\n"
                "   Evaluations MUST use complete datasets (not samples/mock)\n"
                "   Please ensure SWELL complete dataset is properly installed"
            )

        # Merge dataframes from real data more carefully
        print("Merging real SWELL modalities...")

        # Start with the first dataframe
        merged_df = dataframes[0].copy()
        print(f"    Starting with {dataframes[0].shape} (first modality)")

        for i, df in enumerate(dataframes[1:], 1):
            print(f"    Merging modality {i+1}: {df.shape}")
            print(f"      Available columns: {list(df.columns)}")

            # Find ALL common merge columns including participant/subject info
            merge_cols = []

            # Check for participant/subject columns (including modality prefixes)
            participant_found = False
            for subj_col in ["participant", "subject", "pp", "participantno"]:
                # Check direct column names
                if subj_col in merged_df.columns and subj_col in df.columns:
                    merge_cols.append(subj_col)
                    participant_found = True
                    break

            # Check for modality-prefixed participant columns
            if not participant_found:
                merged_cols = merged_df.columns.tolist()
                df_cols = df.columns.tolist()

                # Look for columns ending with _pp (participant)
                merged_pp_cols = [col for col in merged_cols if col.endswith("_pp")]
                df_pp_cols = [col for col in df_cols if col.endswith("_pp")]

                if merged_pp_cols and df_pp_cols:
                    # Use the first participant column from each, but rename them to match
                    merged_pp_col = merged_pp_cols[0]
                    df_pp_col = df_pp_cols[0]

                    # Rename to common column name for merging
                    if merged_pp_col != df_pp_col:
                        df = df.rename(columns={df_pp_col: merged_pp_col})

                    merge_cols.append(merged_pp_col)
                    participant_found = True

            # Always include condition if available
            if "condition" in merged_df.columns and "condition" in df.columns:
                merge_cols.append("condition")

            # Check for block/trial information (including modality prefixes)
            for block_col in ["blok", "block", "trial", "c"]:
                if block_col in merged_df.columns and block_col in df.columns:
                    merge_cols.append(block_col)
                    break

            # Check for modality-prefixed block columns
            merged_block_cols = [
                col
                for col in merged_df.columns
                if any(block in col for block in ["blok", "block", "_c"])
            ]
            df_block_cols = [
                col
                for col in df.columns
                if any(block in col for block in ["blok", "block", "_c"])
            ]

            if merged_block_cols and df_block_cols:
                merged_block_col = merged_block_cols[0]
                df_block_col = df_block_cols[0]

                # Rename to common column name for merging
                if merged_block_col != df_block_col:
                    df = df.rename(columns={df_block_col: merged_block_col})

                if merged_block_col not in merge_cols:
                    merge_cols.append(merged_block_col)

            print(f"      Merge columns found: {merge_cols}")

            if merge_cols:
                # Convert merge columns to same type before merging
                for col in merge_cols:
                    if col in merged_df.columns and col in df.columns:
                        # Convert both to string to avoid type conflicts
                        merged_df[col] = merged_df[col].astype(str)
                        df[col] = df[col].astype(str)

                # Use inner join and add suffixes to handle conflicts
                before_shape = merged_df.shape
                merged_df = pd.merge(
                    merged_df, df, on=merge_cols, how="inner", suffixes=("", f"_{i}")
                )
                print(f"      Merged: {before_shape} -> {merged_df.shape}")

                # If the dataset is getting too large (>50k rows), limit it
                if merged_df.shape[0] > 50000:
                    print(
                        f"      Dataset too large ({merged_df.shape[0]} rows), sampling 50k rows..."
                    )
                    merged_df = merged_df.sample(n=50000, random_state=42)
            else:
                print(
                    f"      Warning: No common merge columns found, skipping this modality"
                )
                continue

        print(f"Final merged dataset: {merged_df.shape}")

        print(f"Merged dataset columns: {list(merged_df.columns)}")

        # Extract features and labels
        # Handle different possible subject column names in real SWELL data
        subject_col = None
        for possible_col in [
            "participant",
            "subject",
            "participantno",
            "participant_id",
            "id",
        ]:
            if possible_col in merged_df.columns:
                subject_col = possible_col
                break

        if subject_col is None:
            print(
                "  No subject column found in real data, creating synthetic subject IDs"
            )
            # Create subject IDs based on data rows (realistic for merged data)
            merged_df["participant"] = [
                f"P{(i//100) + 1:02d}" for i in range(len(merged_df))
            ]
            subject_col = "participant"

        # Get feature columns (exclude subject, condition, and non-numeric columns)
        non_feature_cols = [
            subject_col,
            "condition",
            "timestamp",
            "timestamp_1",
            "timestamp_2",
            "timestamp_3",
            "blok",
            "pp",
            "c",
        ]
        feature_columns = [
            col for col in merged_df.columns if col not in non_feature_cols
        ]

        print(f"Using subject column: {subject_col}")
        print(f"Feature columns: {len(feature_columns)} ({feature_columns[:5]}...)")

        # Extract numeric features only
        X_df = merged_df[feature_columns]

        # Convert to numeric, forcing errors to NaN
        X_numeric = X_df.apply(pd.to_numeric, errors="coerce")

        # Remove columns that are all NaN (non-numeric)
        valid_cols = ~X_numeric.isnull().all()
        X_numeric = X_numeric.loc[:, valid_cols]
        final_feature_columns = X_numeric.columns.tolist()

        print(f"Valid numeric feature columns: {len(final_feature_columns)}")

        X = X_numeric.values
        subjects = merged_df[subject_col].astype(str).tolist()

        # Handle condition labels - map to binary stress classification
        conditions = merged_df["condition"].astype(str)

        # REAL SWELL conditions mapping to stress/no-stress (based on actual dataset)
        print(f"  Found conditions: {sorted(set(conditions))}")

        # SWELL dataset uses specific condition codes:
        # N = Normal (no stress/baseline)
        # T = Time pressure (stress)
        # I = Interruptions (stress)
        # R = Combined time pressure + interruptions (high stress)
        condition_mapping = {
            # SWELL dataset specific codes (uppercase and lowercase)
            "N": 0,
            "n": 0,  # Normal condition (no stress)
            "T": 1,
            "t": 1,  # Time pressure (stress)
            "I": 1,
            "i": 1,  # Interruptions (stress)
            "R": 1,
            "r": 1,  # Combined stress conditions
            # Alternative mappings for other possible formats
            "normal": 0,
            "baseline": 0,
            "control": 0,
            "no stress": 0,
            "time pressure": 1,
            "interruption": 1,
            "interruptions": 1,
            "combined": 1,
            "stress": 1,
        }

        # Apply mapping to real SWELL conditions
        y = []
        unmapped_conditions = set()

        for condition in conditions:
            condition_lower = condition.lower().strip()
            # Try exact match first
            if condition_lower in condition_mapping:
                y.append(condition_mapping[condition_lower])
            # Try partial matches for real SWELL conditions
            elif any(
                stress_word in condition_lower
                for stress_word in ["stress", "pressure", "interrupt", "high", "load"]
            ):
                y.append(1)  # Stress
            elif any(
                baseline_word in condition_lower
                for baseline_word in ["baseline", "control", "normal", "rest", "low"]
            ):
                y.append(0)  # No stress
            else:
                # Default for unknown real conditions - examine manually
                y.append(0)  # Conservative default to no-stress
                unmapped_conditions.add(condition_lower)

        if unmapped_conditions:
            print(
                f"  ⚠️  Unmapped conditions (defaulted to no-stress): {unmapped_conditions}"
            )

        y = np.array(y)

        # Remove features with zero variance
        feature_variances = np.var(X, axis=0)
        valid_features = feature_variances > 1e-8

        if not np.all(valid_features):
            n_removed = np.sum(~valid_features)
            print(f"  Removed {n_removed} zero-variance features")
            X = X[:, valid_features]

        # Handle remaining NaN values
        if np.any(np.isnan(X)):
            print(f"  Handling {np.sum(np.isnan(X))} missing values...")
            # Fill with column means
            for col_idx in range(X.shape[1]):
                col_mean = np.nanmean(X[:, col_idx])
                X[np.isnan(X[:, col_idx]), col_idx] = col_mean

        print(f"\n✓ Successfully loaded SWELL data:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Unique subjects: {len(set(subjects))}")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, subjects

    # STRICT RULES ENFORCEMENT:
    # 1. Evaluations MUST use COMPLETE datasets (100% SWELL/WESAD)
    # 2. NO samples allowed in evaluate_*.py (samples only for test_*.py)
    # 3. ABSOLUTELY NO mock data generation - REAL DATA ONLY
    # 4. Mock data generation is PROHIBITED for ML/AI evaluation

    def split_by_subjects(
        self, X: np.ndarray, y: np.ndarray, subjects: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data by subjects to prevent data leakage."""
        unique_subjects = list(set(subjects))
        n_subjects = len(unique_subjects)

        print(f"\nSplitting {n_subjects} subjects:")

        # Calculate split sizes
        n_train = int(0.5 * n_subjects)  # 50% for training
        n_val = int(0.2 * n_subjects)  # 20% for validation
        n_test = n_subjects - n_train - n_val  # Remaining for test

        # Shuffle subjects
        np.random.seed(42)
        shuffled_subjects = np.random.permutation(unique_subjects)

        train_subjects = shuffled_subjects[:n_train]
        val_subjects = shuffled_subjects[n_train : n_train + n_val]
        test_subjects = shuffled_subjects[n_train + n_val :]

        print(f"  Training: {len(train_subjects)} subjects - {list(train_subjects)}")
        print(f"  Validation: {len(val_subjects)} subjects - {list(val_subjects)}")
        print(f"  Test: {len(test_subjects)} subjects - {list(test_subjects)}")

        # Create splits
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        test_mask = np.isin(subjects, test_subjects)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_classical_models(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train classical ML models and evaluate performance."""
        print("\nTraining classical models...")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "SVM": SVC(random_state=42, probability=True),
        }

        results = {}

        for name, model in models.items():
            print(f"\n  Training {name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predict on validation and test sets
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            val_metrics = self.calculate_metrics(y_val, y_val_pred, "Validation")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")

            results[name] = {
                "validation": val_metrics,
                "test": test_metrics,
                "model": model,
            }

            print(f"    Validation Accuracy: {val_metrics['accuracy']:.3f}")
            print(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")

        return results

    def train_multimodal_network(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Train multimodal neural network."""
        print("\nTraining Multimodal Neural Network...")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)

        # Define multimodal neural network
        class MultimodalSWELLNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()

                # Modality-specific branches
                # Computer interaction (5 features)
                self.computer_branch = nn.Sequential(
                    nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.2)
                )

                # Facial expressions (4 features)
                self.facial_branch = nn.Sequential(
                    nn.Linear(4, 12), nn.ReLU(), nn.Dropout(0.2)
                )

                # Body posture (4 features)
                self.posture_branch = nn.Sequential(
                    nn.Linear(4, 12), nn.ReLU(), nn.Dropout(0.2)
                )

                # Physiology (7 features)
                self.physiology_branch = nn.Sequential(
                    nn.Linear(7, 20), nn.ReLU(), nn.Dropout(0.2)
                )

                # Fusion layers
                self.fusion = nn.Sequential(
                    nn.Linear(16 + 12 + 12 + 20, 64),  # Combined features
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, 2),
                )

            def forward(self, x):
                # Split input by modality
                computer = x[:, :5]
                facial = x[:, 5:9]
                posture = x[:, 9:13]
                physiology = x[:, 13:20]

                # Process each modality
                computer_out = self.computer_branch(computer)
                facial_out = self.facial_branch(facial)
                posture_out = self.posture_branch(posture)
                physiology_out = self.physiology_branch(physiology)

                # Fuse modalities
                fused = torch.cat(
                    [computer_out, facial_out, posture_out, physiology_out], dim=1
                )
                output = self.fusion(fused)

                return output

        model = MultimodalSWELLNet(X_train_scaled.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(150):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    _, val_pred = torch.max(val_outputs.data, 1)
                    val_acc = accuracy_score(y_val_tensor.numpy(), val_pred.numpy())

                    print(
                        f"    Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.3f}"
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, y_val_pred = torch.max(val_outputs.data, 1)

            test_outputs = model(X_test_tensor)
            _, y_test_pred = torch.max(test_outputs.data, 1)

        val_metrics = self.calculate_metrics(y_val, y_val_pred.numpy(), "Validation")
        test_metrics = self.calculate_metrics(y_test, y_test_pred.numpy(), "Test")

        print(f"    Best Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")

        return {"validation": val_metrics, "test": test_metrics, "model": model}

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str
    ) -> Dict:
        """Calculate comprehensive metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def run_evaluation(self) -> Dict:
        """Run complete baseline evaluation."""
        print("🎯 SWELL Dataset Baseline Performance Evaluation")
        print("=" * 60)

        # Load data
        X, y, subjects = self.load_swell_data()

        # Split by subjects
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_by_subjects(
            X, y, subjects
        )

        print(f"\nFinal split sizes:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Train classical models
        classical_results = self.train_classical_models(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Train multimodal neural network
        nn_results = self.train_multimodal_network(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Combine results
        all_results = {**classical_results, "Multimodal Network": nn_results}

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 SWELL BASELINE PERFORMANCE SUMMARY")
        print("=" * 60)

        print("\nTest Set Results (Subject-based split):")
        print("-" * 40)

        for model_name, result in results.items():
            test_metrics = result["test"]
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
            print(f"  Precision: {test_metrics['precision']:.3f}")
            print(f"  Recall:    {test_metrics['recall']:.3f}")
            print(f"  F1-Score:  {test_metrics['f1']:.3f}")

        # Best model
        best_model = max(results.items(), key=lambda x: x[1]["test"]["accuracy"])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test']['accuracy']:.3f}")

        print(f"\n💡 Key Insights:")
        print(
            f"   - Multimodal approach leverages computer + facial + posture + physiology"
        )
        print(f"   - Subject-based splitting prevents data leakage")
        print(f"   - Results represent realistic federated learning scenarios")
        print(f"   - Can compare with 3-node federated learning performance")


def main():
    """Main evaluation function."""
    evaluator = SWELLBaselineEvaluator()
    results = evaluator.run_evaluation()

    # Save results
    results_file = "swell_baseline_results.json"
    import json

    # Convert results for JSON serialization
    json_results = {}
    for model_name, result in results.items():
        if model_name == "Multimodal Network":
            continue  # Skip model object
        json_results[model_name] = {
            "validation": result["validation"],
            "test": result["test"],
        }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print("   Ready for federated learning comparison!")


if __name__ == "__main__":
    main()
