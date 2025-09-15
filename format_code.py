#!/usr/bin/env python3
"""Script to run code formatting and linting."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str, check_mode: bool = False) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=not check_mode, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and not check_mode:
            print(result.stderr)

        if check_mode and result.returncode != 0:
            print(f"⚠️ {description} found issues that need attention")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
        else:
            print(f"✅ {description} completed successfully")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main() -> None:
    """Run code quality tools."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run code formatting and linting")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: don't modify files, just report issues",
    )
    args = parser.parse_args()

    mode_text = "🔍 Verificando" if args.check else "🚀 Ejecutando"
    print(f"{mode_text} herramientas de calidad de código")
    print("=" * 50)

    # Get Python executable from virtual environment
    if os.name == "nt":  # Windows
        python_exe = Path(".venv/Scripts/python.exe")
    else:  # Unix-like
        python_exe = Path(".venv/bin/python")

    if not python_exe.exists():
        python_exe = "python"  # Fallback to system Python

    all_passed = True

    # 1. isort - Sort imports
    isort_cmd = (
        f"{python_exe} -m isort {'--check-only --diff' if args.check else '.'} ."
    )
    if not run_command(isort_cmd, "Ordenando imports con isort", args.check):
        all_passed = False

    # 2. Black - Code formatting
    black_cmd = f"{python_exe} -m black {'--check --diff' if args.check else '.'} ."
    if not run_command(black_cmd, "Formateando código con Black", args.check):
        all_passed = False

    # 3. flake8 - Linting (always in check mode)
    if not run_command(
        f"{python_exe} -m flake8 .", "Ejecutando linting con flake8", True
    ):
        all_passed = False

    # 4. MyPy - Type checking (optional, may have errors)
    print("\n🔄 Verificación de tipos con MyPy...")
    try:
        subprocess.run(f"{python_exe} -m mypy .", shell=True, check=False)
        print("✅ MyPy completado (revisar advertencias arriba)")
    except Exception as e:
        print(f"⚠️ MyPy encontró problemas: {e}")

    print("\n" + "=" * 50)
    if all_passed:
        if args.check:
            print("✅ ¡Código cumple con todos los estándares de calidad!")
        else:
            print("✅ ¡Todas las verificaciones de calidad pasaron!")
        print("\n📋 Resumen:")
        print("   - Imports ordenados con isort")
        print("   - Código formateado con Black")
        print("   - Linting completado con flake8")
        print("   - Verificación de tipos ejecutada")
    else:
        if args.check:
            print("❌ El código necesita correcciones de calidad")
        else:
            print("❌ Algunas verificaciones fallaron")
        print("\n🔧 Pasos recomendados:")
        print("   1. Revisar errores arriba")
        if args.check:
            print("   2. Ejecutar sin --check para aplicar correcciones automáticas")
            print("   3. Corregir problemas manuales restantes")
        else:
            print("   2. Corregir problemas manualmente")
            print("   3. Ejecutar script nuevamente")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
