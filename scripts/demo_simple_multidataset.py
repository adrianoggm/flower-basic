#!/usr/bin/env python3
"""Simple multi-dataset demo using real sample data."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flower_basic.datasets.samples import (
    load_swell_sample_features,
    load_wesad_sample_windows,
)


def train_baseline(X, y, name: str) -> None:
    """Train a logistic regression baseline and print accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} sample accuracy: {accuracy:.3f}")


def main() -> None:
    """Run the lightweight demo with real samples."""
    print("Loading real WESAD sample...")
    X_wesad, y_wesad = load_wesad_sample_windows()
    print(f"WESAD sample windows: {X_wesad.shape}")

    print("\nLoading real SWELL sample...")
    X_swell, y_swell = load_swell_sample_features()
    print(f"SWELL sample rows: {X_swell.shape}")

    print("\nTraining baselines on real samples")
    train_baseline(X_wesad, y_wesad, "WESAD")
    train_baseline(X_swell, y_swell, "SWELL")


if __name__ == "__main__":
    main()
