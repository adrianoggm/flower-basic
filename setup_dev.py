# Development environment setup script
# This script sets up the development environment with all necessary tools

# Python script
"""Development environment setup script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main() -> int:
    """Main setup function."""
    print("🚀 Setting up Federated Learning development environment")
    print("=" * 60)

    # Check Python version

    print(f"✅ Python version: {sys.version}")

    # Install package in development mode
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev,test]"],
        "Installing package with development dependencies",
    ):
        return 1

    # Install pre-commit hooks
    if not run_command(
        [sys.executable, "-m", "pre_commit", "install"], "Installing pre-commit hooks"
    ):
        return 1

    # Run initial quality checks
    print("\n🔍 Running initial quality checks...")

    # Format code
    if not run_command(
        [sys.executable, "-m", "black", "."], "Formatting code with Black"
    ):
        return 1

    if not run_command(
        [sys.executable, "-m", "isort", "."], "Sorting imports with isort"
    ):
        return 1

    # Run linters
    if not run_command(
        [sys.executable, "-m", "ruff", "check", "."], "Running Ruff linter"
    ):
        return 1

    # Type checking
    if not run_command(
        [sys.executable, "-m", "mypy", "."], "Running MyPy type checker"
    ):
        print("⚠️  MyPy found type issues (this is normal for initial setup)")
        print("   Run 'mypy .' to see details")

    # Run tests
    if not run_command(
        [sys.executable, "-m", "pytest", "--collect-only"], "Checking test collection"
    ):
        return 1

    print("\n🎉 Development environment setup completed!")
    print("\n📋 Next steps:")
    print("  • Run 'make test' to execute tests")
    print("  • Run 'make quality' to check code quality")
    print("  • Run 'make run-demo' to start the demo")
    print("  • Run 'pre-commit run --all-files' to check all files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
