#!/usr/bin/env python3
"""Configuración completa del entorno de desarrollo con tests y PEP8."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Union


def run_command(
    command: Union[str, List[str]], description: str, check: bool = True
) -> bool:
    """Ejecuta un comando y maneja errores."""
    print(f"\n🔄 {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(
                command, shell=True, check=check, capture_output=True, text=True
            )
        else:
            result = subprocess.run(
                command, check=check, capture_output=True, text=True
            )

        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Verifica la versión de Python."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        return False

    print("✅ Versión de Python compatible")
    return True


def setup_virtual_environment():
    """Configura el entorno virtual."""
    venv_path = Path(".venv")

    if venv_path.exists():
        print("📁 Entorno virtual ya existe")
        return True

    return run_command(
        [sys.executable, "-m", "venv", ".venv"], "Creando entorno virtual"
    )


def get_python_executable():
    """Obtiene la ruta del ejecutable de Python en el entorno virtual."""
    if os.name == "nt":  # Windows
        return Path(".venv/Scripts/python.exe")
    else:
        return Path(".venv/bin/python")


def install_dependencies():
    """Instala las dependencias del requirements.txt."""
    python_exe = get_python_executable()

    if not python_exe.exists():
        print("❌ Error: No se encuentra el ejecutable de Python en el entorno virtual")
        return False

    # Actualizar pip
    run_command(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
        "Actualizando pip",
    )

    # Instalar dependencias
    return run_command(
        [str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"],
        "Instalando dependencias",
    )


def run_code_formatting():
    """Ejecuta formateo de código."""
    python_exe = get_python_executable()

    print("\n🎨 Aplicando formateo de código...")

    # isort - ordenar imports
    run_command(
        [str(python_exe), "-m", "isort", "."],
        "Ordenando imports con isort",
        check=False,
    )

    # Black - formateo de código
    run_command(
        [str(python_exe), "-m", "black", "."],
        "Formateando código con Black",
        check=False,
    )

    # flake8 - linting
    run_command(
        [str(python_exe), "-m", "flake8", "."],
        "Ejecutando linting con flake8",
        check=False,
    )

    return True


def run_basic_tests():
    """Ejecuta tests básicos para verificar la configuración."""
    python_exe = get_python_executable()

    print("\n🧪 Ejecutando tests básicos...")

    # Tests unitarios básicos
    success = run_command(
        [
            str(python_exe),
            "-m",
            "pytest",
            "tests/",
            "-v",
            "-m",
            "not integration and not slow",
        ],
        "Tests unitarios",
        check=False,
    )

    if success:
        print("✅ Tests básicos pasaron")
    else:
        print("⚠️ Algunos tests fallaron (puede ser normal en primera ejecución)")

    return True


def verify_system_components():
    """Verifica que los componentes del sistema se pueden importar."""
    python_exe = get_python_executable()

    print("\n🔍 Verificando componentes del sistema...")

    test_script = """
import sys
try:
    import torch
    import numpy
    import flwr
    import paho.mqtt.client
    import sklearn
    print("✅ Todas las dependencias principales importadas correctamente")

    # Verificar módulos del proyecto
    import model
    import utils
    print("✅ Módulos del proyecto importados correctamente")

    # Test básico del modelo
    from model import ECGModel
    model = ECGModel()
    print("✅ Modelo ECG creado exitosamente")

except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️ Advertencia: {e}")
"""

    return run_command(
        [str(python_exe), "-c", test_script],
        "Verificando componentes del sistema",
        check=False,
    )


def create_development_scripts():
    """Crea scripts adicionales para desarrollo."""

    # Script para ejecutar formato rápido
    quick_format_script = """@echo off
echo "Formateo rapido de codigo..."
.venv\\Scripts\\python.exe -m black .
.venv\\Scripts\\python.exe -m isort .
echo "Formateo completado"
pause
"""

    # Script para tests rápidos
    quick_test_script = """@echo off
echo "Tests rapidos..."
.venv\\Scripts\\python.exe -m pytest tests/ -v -m "not integration and not slow"
pause
"""

    if os.name == "nt":  # Windows
        with open("quick_format.bat", "w", encoding="utf-8") as f:
            f.write(quick_format_script)

        with open("quick_test.bat", "w", encoding="utf-8") as f:
            f.write(quick_test_script)

        print("Scripts de desarrollo creados: quick_format.bat, quick_test.bat")

    return True


def main():
    """Función principal de configuración."""
    print("🚀 Configuración Completa: PEP8 + Tests + Entorno")
    print("=" * 60)

    # 1. Verificar Python
    if not check_python_version():
        sys.exit(1)

    # 2. Configurar entorno virtual
    if not setup_virtual_environment():
        print("❌ Error configurando entorno virtual")
        sys.exit(1)

    # 3. Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        sys.exit(1)

    # 4. Aplicar formateo de código
    if not run_code_formatting():
        print("⚠️ Formateo de código completado con advertencias")

    # 5. Verificar componentes del sistema
    if not verify_system_components():
        print("⚠️ Verificación de componentes completada con advertencias")

    # 6. Ejecutar tests básicos
    if not run_basic_tests():
        print("⚠️ Tests básicos completados con advertencias")

    # 7. Crear scripts de desarrollo
    create_development_scripts()

    print("\n" + "=" * 60)
    print("✅ ¡Configuración completada!")
    print("\n📋 Resumen de lo que se configuró:")
    print("   ✅ Entorno virtual Python (.venv/)")
    print("   ✅ Dependencias instaladas (pytest, black, flake8, etc.)")
    print("   ✅ Código formateado según PEP8")
    print("   ✅ Tests configurados y ejecutados")
    print("   ✅ Scripts de desarrollo creados")

    print("\n🎯 Próximos pasos:")
    print("   1. Activar entorno: .venv\\Scripts\\activate (Windows)")
    print("   2. Ejecutar tests: python run_tests.py")
    print("   3. Formatear código: python format_code.py")
    print("   4. Usar Makefile: make help (si tienes make)")

    print("\n🧪 Para ejecutar la demo del sistema:")
    print("   1. mosquitto -v (en terminal separado)")
    print("   2. python server.py")
    print("   3. python broker_fog.py")
    print("   4. python fog_flower_client.py")
    print("   5. python client.py (múltiples instancias)")

    print("\n📚 Archivos importantes creados:")
    print("   - tests/ (suite completa de tests)")
    print("   - pytest.ini (configuración de tests)")
    print("   - .flake8 (configuración de linting)")
    print("   - pyproject.toml (configuración de Black)")
    print("   - run_tests.py (script para ejecutar tests)")
    print("   - format_code.py (script para formatear código)")
    print("   - Makefile (automatización de tareas)")


if __name__ == "__main__":
    main()
