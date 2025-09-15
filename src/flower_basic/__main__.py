"""Command-line interface for Federated Learning with Fog Computing Demo.

This module provides a modern CLI interface for running various components
of the federated learning system with proper argument parsing and help.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from .compare_models import ModelComparator


def main() -> None:
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        prog="flower-basic",
        description="Federated Learning with Fog Computing Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s compare --cv-folds 5        # Run robust comparison
  %(prog)s demo                        # Run quick demo
  %(prog)s --help                      # Show this help
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare federated vs centralized learning",
        description="Run comprehensive comparison between federated and centralized approaches",
    )
    compare_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    compare_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    compare_parser.add_argument(
        "--clients",
        type=int,
        default=3,
        help="Number of federated clients (default: 3)",
    )
    compare_parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results"),
        help="Output directory for results (default: comparison_results)",
    )

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run quick demonstration",
        description="Run a quick demonstration of the federated learning system",
    )
    demo_parser.add_argument(
        "--fast",
        action="store_true",
        help="Run faster demo with reduced parameters",
    )

    # Info command
    subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display information about the system and dependencies",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "compare":
            run_comparison(args)
        elif args.command == "demo":
            run_demo(args)
        elif args.command == "info":
            show_info()
        else:
            parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_comparison(args: argparse.Namespace) -> None:
    """Run model comparison with specified parameters."""
    print("🚀 Starting robust model comparison...")
    print(f"   Cross-validation folds: {args.cv_folds}")
    print(f"   Training epochs: {args.epochs}")
    print(f"   Federated clients: {args.clients}")
    print(f"   FL rounds: {args.rounds}")
    print(f"   Output directory: {args.output_dir}")
    print()

    comparator = ModelComparator()
    results = comparator.run_robust_comparison(
        epochs=args.epochs,
        num_clients=args.clients,
        fl_rounds=args.rounds,
        n_cv_folds=args.cv_folds,
    )

    print("✅ Comparison completed successfully!")
    print(f"📊 Results saved to: {args.output_dir}/robust_comparison_results.json")

    # Print key findings
    cv_results = results["cross_validation"]
    stat_results = results["statistical_test"]

    print("\n📈 Key Results:")
    print(".4f")
    print(".4f")
    print(".3f")
    print(f"   Cohen's d: {stat_results['cohen_d']:.3f}")
    print(f"   Significant: {'Yes' if stat_results['significant'] else 'No'}")

    if results["data_leakage"]["leakage_detected"]:
        print("\n⚠️  Data leakage detected!")
        print(".1f")


def run_demo(args: argparse.Namespace) -> None:
    """Run quick demonstration."""
    print("🎯 Running federated learning demo...")

    if args.fast:
        print("   Mode: Fast (reduced parameters)")
        # Run quick comparison
        import subprocess
        result = subprocess.run([
            sys.executable, "quick_comparison.py"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed!")
            print(result.stderr)
            sys.exit(1)
    else:
        print("   Mode: Standard")
        # Run standard comparison
        comparator = ModelComparator()
        results = comparator.run_robust_comparison(n_cv_folds=2)

        print("✅ Demo completed successfully!")
        print("📊 Results saved to: comparison_results/robust_comparison_results.json")


def show_info() -> None:
    """Show system information."""
    import platform
    import torch

    print("🖥️  System Information")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("CUDA version: Available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n📦 Package Information")
    print("=" * 50)

    try:
        import flwr
        print(f"Flower version: {flwr.__version__}")
    except ImportError:
        print("Flower: Not installed")

    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")

    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("Pandas: Not installed")

    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn: Not installed")


if __name__ == "__main__":
    main()
