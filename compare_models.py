"""Performance comparison between federated and centralized models.

This script runs both baseline (centralized) and federated training approaches
on the same data and generates a comprehensive comparison report.

Usage:
    python compare_models.py [--epochs 50] [--num_clients 3] [--fl_rounds 10]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from baseline_model import BaselineTrainer
from model import ECGModel
from utils import load_ecg5000_openml


class FederatedSimulator:
    """Simple federated learning simulator for comparison."""

    def __init__(
        self, num_clients: int = 3, device: torch.device = torch.device("cpu")
    ):
        """Initialize federated learning simulator.

        Args:
            num_clients: Number of federated clients
            device: Device to run on
        """
        self.num_clients = num_clients
        self.device = device
        self.global_model = ECGModel().to(device)
        self.client_models = [ECGModel().to(device) for _ in range(num_clients)]

    def split_data_federated(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data among federated clients.

        Args:
            X: Features
            y: Labels

        Returns:
            List of (X_client, y_client) tuples
        """
        # Simple random split (IID)
        indices = np.random.permutation(len(X))
        split_size = len(X) // self.num_clients

        client_data = []
        for i in range(self.num_clients):
            start_idx = i * split_size
            if i == self.num_clients - 1:  # Last client gets remaining data
                end_idx = len(X)
            else:
                end_idx = (i + 1) * split_size

            client_indices = indices[start_idx:end_idx]
            client_data.append((X[client_indices], y[client_indices]))

        return client_data

    def federated_averaging(self) -> None:
        """Perform federated averaging of client models."""
        global_state = self.global_model.state_dict()

        # Average parameters from all clients
        for key in global_state.keys():
            # Stack all client parameters for this layer
            client_params = torch.stack(
                [client.state_dict()[key] for client in self.client_models]
            )
            # Average them
            global_state[key] = torch.mean(client_params, dim=0)

        # Update global model
        self.global_model.load_state_dict(global_state)

        # Update all client models with new global parameters
        for client in self.client_models:
            client.load_state_dict(global_state)

    def train_federated(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rounds: int = 10,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, Any]:
        """Simulate federated training.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            rounds: Number of federated rounds
            local_epochs: Local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            Dictionary with training results
        """
        print(
            f"Starting federated training ({self.num_clients} clients, {rounds} rounds)..."
        )

        # Split data among clients
        client_data = self.split_data_federated(X_train, y_train)

        # Create trainers for each client
        client_trainers = []
        for i, (X_client, y_client) in enumerate(client_data):
            trainer = BaselineTrainer(
                model=self.client_models[i],
                device=self.device,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            train_loader, _ = trainer.create_data_loaders(
                X_client, y_client, X_test, y_test
            )
            client_trainers.append((trainer, train_loader))
            print(f"   Client {i+1}: {len(X_client)} samples")

        # Create test loader for global evaluation
        global_trainer = BaselineTrainer(
            model=self.global_model,
            device=self.device,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        _, test_loader = global_trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        # Training history
        round_metrics = []
        start_time = time.time()

        for round_num in range(rounds):
            print(f"Round {round_num + 1}/{rounds}")

            # Local training on each client
            for client_id, (trainer, train_loader) in enumerate(client_trainers):
                for epoch in range(local_epochs):
                    trainer.train_epoch(train_loader)

            # Federated averaging
            self.federated_averaging()

            # Evaluate global model
            metrics = global_trainer.evaluate(test_loader)
            metrics["round"] = round_num + 1
            round_metrics.append(metrics)

            if (round_num + 1) % 2 == 0:
                print(f"   Global Accuracy: {metrics['accuracy']:.4f}")

        training_time = time.time() - start_time
        print(f"Federated training completed in {training_time:.2f} seconds")

        return {
            "final_metrics": round_metrics[-1],
            "round_metrics": round_metrics,
            "training_time": training_time,
            "num_clients": self.num_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
        }


class ModelComparator:
    """Compare federated and centralized model performance."""

    def __init__(self, output_dir: str = "comparison_results"):
        """Initialize comparator.

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_comparison(
        self,
        epochs: int = 50,
        num_clients: int = 3,
        fl_rounds: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Run complete comparison between models.

        Args:
            epochs: Epochs for centralized training
            num_clients: Number of federated clients
            fl_rounds: Federated learning rounds
            batch_size: Batch size
            learning_rate: Learning rate
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Comparison results dictionary
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("*** Model Performance Comparison ***")
        print("=" * 60)

        # Load data
        print("Loading ECG5000 dataset...")
        X_train, X_test, y_train, y_test = load_ecg5000_openml(
            test_size=test_size, random_state=random_state
        )

        print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")

        # 1. Baseline (Centralized) Training
        print("\n1. Running Baseline (Centralized) Training...")
        baseline_model = ECGModel().to(device)
        baseline_trainer = BaselineTrainer(
            model=baseline_model,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        train_loader, test_loader = baseline_trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        baseline_results = baseline_trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            verbose=False,
        )

        print(
            f"Baseline completed: Accuracy = {baseline_results['final_metrics']['accuracy']:.4f}"
        )

        # 2. Federated Training
        print(f"\n2. Running Federated Training...")
        fed_simulator = FederatedSimulator(num_clients=num_clients, device=device)

        federated_results = fed_simulator.train_federated(
            X_train,
            y_train,
            X_test,
            y_test,
            rounds=fl_rounds,
            local_epochs=epochs // fl_rounds,  # Approximate same total training
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        print(
            f"Federated completed: Accuracy = {federated_results['final_metrics']['accuracy']:.4f}"
        )

        # 3. Generate comparison
        comparison = self.generate_comparison_report(
            baseline_results, federated_results
        )

        # 4. Save results
        self.save_comparison_results(comparison, baseline_results, federated_results)

        return comparison

    def generate_comparison_report(
        self, baseline_results: Dict[str, Any], federated_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive comparison report.

        Args:
            baseline_results: Results from baseline training
            federated_results: Results from federated training

        Returns:
            Comparison dictionary
        """
        baseline_metrics = baseline_results["final_metrics"]
        federated_metrics = federated_results["final_metrics"]

        # Calculate differences
        accuracy_diff = federated_metrics["accuracy"] - baseline_metrics["accuracy"]
        f1_diff = federated_metrics["f1"] - baseline_metrics["f1"]
        time_ratio = (
            federated_results["training_time"] / baseline_results["training_time"]
        )

        comparison = {
            "baseline": {
                "accuracy": baseline_metrics["accuracy"],
                "f1_score": baseline_metrics["f1"],
                "precision": baseline_metrics["precision"],
                "recall": baseline_metrics["recall"],
                "auc": baseline_metrics["auc"],
                "training_time": baseline_results["training_time"],
                "epochs": baseline_results["epochs"],
            },
            "federated": {
                "accuracy": federated_metrics["accuracy"],
                "f1_score": federated_metrics["f1"],
                "precision": federated_metrics["precision"],
                "recall": federated_metrics["recall"],
                "auc": federated_metrics["auc"],
                "training_time": federated_results["training_time"],
                "rounds": federated_results["rounds"],
                "clients": federated_results["num_clients"],
            },
            "differences": {
                "accuracy_diff": accuracy_diff,
                "accuracy_diff_pct": (accuracy_diff / baseline_metrics["accuracy"])
                * 100,
                "f1_diff": f1_diff,
                "f1_diff_pct": (
                    (f1_diff / baseline_metrics["f1"]) * 100
                    if baseline_metrics["f1"] > 0
                    else 0
                ),
                "time_ratio": time_ratio,
                "time_overhead_pct": (time_ratio - 1) * 100,
            },
            "summary": {
                "better_approach": "federated" if accuracy_diff > 0 else "centralized",
                "accuracy_gap": abs(accuracy_diff),
                "training_efficiency": "federated" if time_ratio < 1 else "centralized",
            },
        }

        return comparison

    def save_comparison_results(
        self,
        comparison: Dict[str, Any],
        baseline_results: Dict[str, Any],
        federated_results: Dict[str, Any],
    ) -> None:
        """Save comparison results and generate visualizations.

        Args:
            comparison: Comparison results
            baseline_results: Baseline training results
            federated_results: Federated training results
        """
        # Save JSON report
        with open(os.path.join(self.output_dir, "comparison_report.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        # Generate text report
        self.generate_text_report(comparison)

        # Generate comparison plots
        self.plot_comparison(comparison, baseline_results, federated_results)

        print(f"\nComparison results saved to {self.output_dir}/")

    def generate_text_report(self, comparison: Dict[str, Any]) -> None:
        """Generate human-readable text report.

        Args:
            comparison: Comparison results
        """
        report_path = os.path.join(self.output_dir, "comparison_report.txt")

        with open(report_path, "w") as f:
            f.write("*** ECG Model Performance Comparison Report ***\n")
            f.write("=" * 50 + "\n\n")

            # Performance comparison table
            f.write("Performance Metrics Comparison\n")
            f.write("-" * 40 + "\n")

            metrics_data = [
                ["Metric", "Centralized", "Federated", "Difference"],
                [
                    "Accuracy",
                    f"{comparison['baseline']['accuracy']:.4f}",
                    f"{comparison['federated']['accuracy']:.4f}",
                    f"{comparison['differences']['accuracy_diff']:+.4f}",
                ],
                [
                    "F1-Score",
                    f"{comparison['baseline']['f1_score']:.4f}",
                    f"{comparison['federated']['f1_score']:.4f}",
                    f"{comparison['differences']['f1_diff']:+.4f}",
                ],
                [
                    "Precision",
                    f"{comparison['baseline']['precision']:.4f}",
                    f"{comparison['federated']['precision']:.4f}",
                    "-",
                ],
                [
                    "Recall",
                    f"{comparison['baseline']['recall']:.4f}",
                    f"{comparison['federated']['recall']:.4f}",
                    "-",
                ],
                [
                    "AUC",
                    f"{comparison['baseline']['auc']:.4f}",
                    f"{comparison['federated']['auc']:.4f}",
                    "-",
                ],
            ]

            f.write(tabulate(metrics_data, headers="firstrow", tablefmt="grid"))
            f.write("\n\n")

            # Training efficiency
            f.write("Training Efficiency\n")
            f.write("-" * 25 + "\n")
            f.write(
                f"Centralized Training Time: {comparison['baseline']['training_time']:.2f}s\n"
            )
            f.write(
                f"Federated Training Time: {comparison['federated']['training_time']:.2f}s\n"
            )
            f.write(
                f"Time Ratio (Fed/Central): {comparison['differences']['time_ratio']:.2f}x\n"
            )
            f.write(
                f"Time Overhead: {comparison['differences']['time_overhead_pct']:+.1f}%\n\n"
            )

            # Summary
            f.write("Summary\n")
            f.write("-" * 12 + "\n")
            f.write(
                f"Better Accuracy: {comparison['summary']['better_approach'].title()}\n"
            )
            f.write(f"Accuracy Gap: {comparison['summary']['accuracy_gap']:.4f}\n")
            f.write(
                f"More Efficient: {comparison['summary']['training_efficiency'].title()}\n"
            )

            # Recommendations
            f.write("\nRecommendations\n")
            f.write("-" * 20 + "\n")

            if (
                comparison["differences"]["accuracy_diff"] > -0.02
            ):  # Within 2% is acceptable
                f.write("GOOD: Federated learning maintains competitive accuracy\n")
            else:
                f.write(
                    "WARNING: Significant accuracy degradation in federated approach\n"
                )

            if comparison["differences"]["time_ratio"] < 1.5:  # Less than 50% overhead
                f.write("GOOD: Federated learning has reasonable training overhead\n")
            else:
                f.write("WARNING: High training overhead in federated approach\n")

    def plot_comparison(
        self,
        comparison: Dict[str, Any],
        baseline_results: Dict[str, Any],
        federated_results: Dict[str, Any],
    ) -> None:
        """Generate comparison plots.

        Args:
            comparison: Comparison results
            baseline_results: Baseline training results
            federated_results: Federated training results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Performance metrics comparison
        metrics = ["Accuracy", "F1-Score", "Precision", "Recall", "AUC"]
        baseline_vals = [
            comparison["baseline"]["accuracy"],
            comparison["baseline"]["f1_score"],
            comparison["baseline"]["precision"],
            comparison["baseline"]["recall"],
            comparison["baseline"]["auc"],
        ]
        federated_vals = [
            comparison["federated"]["accuracy"],
            comparison["federated"]["f1_score"],
            comparison["federated"]["precision"],
            comparison["federated"]["recall"],
            comparison["federated"]["auc"],
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width / 2, baseline_vals, width, label="Centralized", alpha=0.8)
        ax1.bar(x + width / 2, federated_vals, width, label="Federated", alpha=0.8)
        ax1.set_xlabel("Metrics")
        ax1.set_ylabel("Score")
        ax1.set_title("Performance Metrics Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Training time comparison
        times = [
            comparison["baseline"]["training_time"],
            comparison["federated"]["training_time"],
        ]
        approaches = ["Centralized", "Federated"]
        colors = ["skyblue", "lightcoral"]

        bars = ax2.bar(approaches, times, color=colors, alpha=0.8)
        ax2.set_ylabel("Training Time (seconds)")
        ax2.set_title("Training Time Comparison")
        ax2.grid(True, alpha=0.3)

        # Add values on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
            )

        # 3. Training curves (if available)
        if "train_accuracies" in baseline_results:
            epochs = range(1, len(baseline_results["train_accuracies"]) + 1)
            ax3.plot(
                epochs,
                baseline_results["train_accuracies"],
                "b-",
                label="Centralized Train",
            )
            ax3.plot(
                epochs,
                baseline_results["val_accuracies"],
                "b--",
                label="Centralized Val",
            )

        if "round_metrics" in federated_results:
            rounds = [m["round"] for m in federated_results["round_metrics"]]
            fed_accs = [m["accuracy"] for m in federated_results["round_metrics"]]
            ax3.plot(rounds, fed_accs, "r-", label="Federated Global", marker="o")

        ax3.set_xlabel("Epoch/Round")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Training Progress")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Accuracy difference visualization
        diff_pct = comparison["differences"]["accuracy_diff_pct"]
        color = "green" if diff_pct >= 0 else "red"

        ax4.bar(["Accuracy Difference"], [diff_pct], color=color, alpha=0.7)
        ax4.set_ylabel("Difference (%)")
        ax4.set_title("Federated vs Centralized Accuracy")
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax4.grid(True, alpha=0.3)

        # Add value on bar
        ax4.text(
            0,
            diff_pct + (0.1 if diff_pct >= 0 else -0.1),
            f"{diff_pct:+.2f}%",
            ha="center",
            va="bottom" if diff_pct >= 0 else "top",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "comparison_plots.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Main function to run model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare federated vs centralized ECG models"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Epochs for centralized training"
    )
    parser.add_argument(
        "--num_clients", type=int, default=3, help="Number of federated clients"
    )
    parser.add_argument(
        "--fl_rounds", type=int, default=10, help="Federated learning rounds"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="comparison_results", help="Output directory"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # Run comparison
    comparator = ModelComparator(args.output_dir)

    results = comparator.run_comparison(
        epochs=args.epochs,
        num_clients=args.num_clients,
        fl_rounds=args.fl_rounds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nModel comparison completed successfully!")
    print(f"Check {args.output_dir}/ for detailed results and visualizations")


if __name__ == "__main__":
    main()
