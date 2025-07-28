"""Quick model comparison demo.

This script runs a quick comparison between federated and centralized approaches
with reduced parameters for fast testing.

Usage:
    python quick_comparison.py
"""

import subprocess
import sys
import time


def run_quick_comparison():
    """Run quick comparison with reduced parameters."""
    print("*** Quick ECG Model Comparison Demo ***")
    print("=" * 50)
    print("Configuration:")
    print("   • Centralized: 20 epochs")
    print("   • Federated: 3 clients, 5 rounds")
    print("   • Batch size: 16 (smaller for speed)")
    print("   • Learning rate: 0.001")
    print("-" * 50)

    start_time = time.time()

    try:
        # Run quick comparison
        cmd = [
            sys.executable,
            "compare_models.py",
            "--epochs",
            "20",
            "--num_clients",
            "3",
            "--fl_rounds",
            "5",
            "--batch_size",
            "16",
            "--lr",
            "0.001",
            "--output_dir",
            "quick_comparison_results",
        ]

        print("Running comparison...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            total_time = time.time() - start_time
            print(f"\nQuick comparison completed in {total_time:.1f} seconds!")
            print(f"Results saved to: quick_comparison_results/")
            print(f"Check comparison_plots.png for visualizations")
            print(f"Check comparison_report.txt for detailed analysis")

            # Try to show quick summary from stdout
            if "Better Accuracy:" in result.stdout:
                lines = result.stdout.split("\n")
                for line in lines:
                    if any(
                        keyword in line
                        for keyword in ["Accuracy =", "completed:", "Better"]
                    ):
                        print(f"   {line.strip()}")
        else:
            print(f"Error running comparison:")
            print(result.stderr)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    run_quick_comparison()
