"""Baseline centralized model for performance comparison.

This script trains the same ECGModel architecture used in federated learning
but in a traditional centralized manner. This allows for direct performance
comparison between federated and centralized approaches.

Usage:
    python baseline_model.py [--epochs 50] [--batch_size 32] [--lr 0.001]
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .datasets import load_wesad_dataset
from .model import ECGModel


class BaselineTrainer:
    """Centralized trainer for ECG model baseline comparison."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        """Initialize the baseline trainer.

        Args:
            model: ECG model to train
            device: Device to run training on
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        # Use same loss and optimizer as federated setup
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Track training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders from numpy arrays.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (train_loader, test_loader)
        """
        try:
            # Handle empty datasets
            if len(X_train) == 0 or len(y_train) == 0:
                # Create empty tensors with correct shape
                X_train_tensor = torch.empty(0, 1, 140, dtype=torch.float32)
                y_train_tensor = torch.empty(0, 1, dtype=torch.float32)
            else:
                # Convert to tensors and reshape for CNN (add channel dimension)
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

            if len(X_test) == 0 or len(y_test) == 0:
                X_test_tensor = torch.empty(0, 1, 140, dtype=torch.float32)
                y_test_tensor = torch.empty(0, 1, dtype=torch.float32)
            else:
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )

            return train_loader, test_loader
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            # Return empty loaders as fallback
            empty_dataset = TensorDataset(torch.empty(0, 1, 140), torch.empty(0, 1))
            empty_loader = DataLoader(empty_dataset, batch_size=self.batch_size)
            return empty_loader, empty_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                # Get predictions and probabilities
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                y_true.extend(batch_y.cpu().numpy().flatten())
                y_pred.extend(predicted.cpu().numpy().flatten())
                y_prob.extend(probs.cpu().numpy().flatten())

        # Calculate comprehensive metrics
        metrics = {
            "loss": total_loss / len(test_loader),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the model for specified epochs.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Dictionary with training history and final metrics
        """
        start_time = time.time()

        if verbose:
            print(f"ð Starting centralized training for {epochs} epochs...")
            print(
                f"ð Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )
            print(f"ð§ Device: {self.device}")
            print("-" * 60)

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluate(test_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

        training_time = time.time() - start_time

        # Final evaluation
        final_metrics = self.evaluate(test_loader)

        if verbose:
            print("-" * 60)
            print(f"â Training completed in {training_time:.2f} seconds")
            print("ð Final Test Metrics:")
            for metric, value in final_metrics.items():
                print(f"   {metric.capitalize()}: {value:.4f}")

        return {
            "final_metrics": final_metrics,
            "training_time": training_time,
            "epochs": epochs,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def save_results(
        self, results: Dict[str, Any], output_dir: str = "baseline_results"
    ):
        """Save training results and model.

        Args:
            results: Results dictionary from training
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics as JSON
        metrics_file = os.path.join(output_dir, "baseline_metrics.json")

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                json_results[key] = (
                    list(value) if isinstance(value, np.ndarray) else value
                )
            else:
                json_results[key] = value

        with open(metrics_file, "w") as f:
            json.dump(json_results, f, indent=2)

        # Save model
        model_file = os.path.join(output_dir, "baseline_model.pth")
        torch.save(self.model.state_dict(), model_file)

        # Create training plots
        self.plot_training_history(output_dir)

        print(f"ð¾ Results saved to {output_dir}/")
        print(f"   ð Metrics: {metrics_file}")
        print(f"   ð§  Model: {model_file}")
        print(f"   ð Plots: {output_dir}/training_plots.png")

    def plot_training_history(self, output_dir: str):
        """Plot training history.

        Args:
            output_dir: Directory to save plots
        """
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, "b-", label="Training Loss")
        ax1.plot(epochs, self.val_losses, "r-", label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, self.train_accuracies, "b-", label="Training Accuracy")
        ax2.plot(epochs, self.val_accuracies, "r-", label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "training_plots.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main function to run baseline training."""
    parser = argparse.ArgumentParser(description="Train baseline centralized ECG model")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="baseline_results", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: cpu, cuda, or auto"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("ð¥ Baseline ECG Model Training")
    print(f"ð+/- Device: {device}")
    print("âï¸  Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Test Size: {args.test_size}")
    print("=" * 60)

    # Load data
    print("ð¥ Loading WESAD dataset...")
    X_train, X_test, y_train, y_test = load_wesad_dataset(
        test_size=args.test_size, random_state=args.random_state
    )

    print("ð Dataset info:")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Positive class ratio: {y_train.mean():.3f}")

    # Initialize model and trainer
    model = ECGModel(in_channels=1, seq_len=X_train.shape[1])
    trainer = BaselineTrainer(
        model=model, device=device, learning_rate=args.lr, batch_size=args.batch_size
    )

    # Create data loaders
    train_loader, test_loader = trainer.create_data_loaders(
        X_train, y_train, X_test, y_test
    )

    # Train model
    results = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        verbose=True,
    )

    # Save results
    trainer.save_results(results, args.output_dir)

    print("\nð Baseline training completed successfully!")
    print("ð Use these results to compare with federated learning performance.")


if __name__ == "__main__":
    main()
