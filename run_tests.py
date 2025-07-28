#!/usr/bin/env python3
"""Script to run tests with different configurations."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False


def main():
    """Run different test configurations."""
    print("🧪 Ejecutando Suite de Tests del Sistema")
    print("=" * 50)

    # Get Python executable from virtual environment
    if os.name == "nt":  # Windows
        python_exe = Path(".venv/Scripts/python.exe")
    else:  # Unix-like
        python_exe = Path(".venv/bin/python")

    if not python_exe.exists():
        python_exe = "python"  # Fallback to system Python

    test_configs = [
        # Basic unit tests
        {
            "command": f"{python_exe} -m pytest tests/ -v -m 'not integration and not slow'",
            "description": "Tests unitarios básicos",
        },
        # Unit tests with coverage
        {
            "command": f"{python_exe} -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html -m 'not integration and not slow'",
            "description": "Tests unitarios con cobertura",
        },
        # Integration tests (if MQTT broker available)
        {
            "command": f"{python_exe} -m pytest tests/ -v -m integration",
            "description": "Tests de integración (requiere MQTT broker)",
        },
        # All tests
        {
            "command": f"{python_exe} -m pytest tests/ -v",
            "description": "Todos los tests",
        },
    ]

    print("Selecciona qué tests ejecutar:")
    for i, config in enumerate(test_configs, 1):
        print(f"   {i}. {config['description']}")
    print("   5. Ejecutar todos en secuencia")

    try:
        choice = input("\nIngresa tu opción (1-5): ").strip()

        if choice == "5":
            # Run all configurations
            all_passed = True
            for config in test_configs:
                if not run_command(config["command"], config["description"]):
                    all_passed = False

            print("\n" + "=" * 50)
            if all_passed:
                print("✅ ¡Todos los tests pasaron!")
            else:
                print("❌ Algunos tests fallaron")

        elif choice in ["1", "2", "3", "4"]:
            config = test_configs[int(choice) - 1]
            success = run_command(config["command"], config["description"])

            print("\n" + "=" * 50)
            if success:
                print("✅ Tests completados exitosamente!")
            else:
                print("❌ Tests fallaron")

        else:
            print("❌ Opción inválida")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrumpidos por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error ejecutando tests: {e}")
        return 1

    # Additional test information
    print("\n📋 Información adicional:")
    print("   - Reports de cobertura en htmlcov/")
    print("   - Tests marcados como 'integration' requieren MQTT broker")
    print("   - Tests marcados como 'slow' pueden tomar más tiempo")
    print(
        "   - Para tests individuales: pytest tests/test_model.py::TestECGModel::test_forward_pass"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
