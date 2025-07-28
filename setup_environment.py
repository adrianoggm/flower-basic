#!/usr/bin/env python3
"""
Script de configuración automática del entorno para Federated Fog Demo
Configura el entorno virtual, instala dependencias y verifica la configuración
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description, check=True):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(f"✅ Éxito: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        return False
    
    print("✅ Versión de Python compatible")
    return True

def setup_virtual_environment():
    """Configura el entorno virtual"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("📁 Entorno virtual ya existe")
        return True
    
    # Crear entorno virtual
    if not run_command([sys.executable, "-m", "venv", ".venv"], 
                      "Creando entorno virtual"):
        return False
    
    return True

def get_python_executable():
    """Obtiene la ruta del ejecutable de Python en el entorno virtual"""
    if platform.system() == "Windows":
        return Path(".venv/Scripts/python.exe")
    else:
        return Path(".venv/bin/python")

def install_dependencies():
    """Instala las dependencias del requirements.txt"""
    python_exe = get_python_executable()
    
    if not python_exe.exists():
        print("❌ Error: No se encuentra el ejecutable de Python en el entorno virtual")
        return False
    
    # Actualizar pip
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
               "Actualizando pip")
    
    # Instalar dependencias
    return run_command([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"],
                      "Instalando dependencias")

def verify_installation():
    """Verifica que las dependencias están correctamente instaladas"""
    python_exe = get_python_executable()
    
    packages = ["torch", "flwr", "paho-mqtt", "numpy", "scikit-learn"]
    
    print("\n🔍 Verificando instalación de paquetes...")
    all_good = True
    
    for package in packages:
        cmd = [str(python_exe), "-c", f"import {package}; print(f'{package}: OK')"]
        if not run_command(cmd, f"Verificando {package}", check=False):
            all_good = False
    
    return all_good

def test_mqtt_connection():
    """Prueba la conexión al broker MQTT"""
    python_exe = get_python_executable()
    
    test_script = '''
import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Conexión MQTT exitosa")
        client.disconnect()
    else:
        print(f"❌ Error de conexión MQTT: {rc}")

client = mqtt.Client()
client.on_connect = on_connect
try:
    client.connect("test.mosquitto.org", 1883, 5)
    client.loop_start()
    time.sleep(2)
    client.loop_stop()
except Exception as e:
    print(f"❌ Error de conexión: {e}")
'''
    
    return run_command([str(python_exe), "-c", test_script],
                      "Probando conexión MQTT", check=False)

def create_run_scripts():
    """Crea scripts de ejecución para Windows"""
    if platform.system() != "Windows":
        return True
    
    python_exe = get_python_executable()
    
    scripts = {
        "run_server.bat": f'@echo off\necho "Iniciando servidor central..."\n{python_exe} server.py\npause',
        "run_fog_broker.bat": f'@echo off\necho "Iniciando fog broker..."\n{python_exe} broker_fog.py\npause',
        "run_fog_client.bat": f'@echo off\necho "Iniciando fog client..."\n{python_exe} fog_flower_client.py\npause',
        "run_client.bat": f'@echo off\necho "Iniciando cliente local..."\n{python_exe} client.py\npause',
        "run_debug.bat": f'@echo off\necho "Iniciando monitor MQTT..."\n{python_exe} debug.py\npause'
    }
    
    for filename, content in scripts.items():
        with open(filename, 'w') as f:
            f.write(content)
    
    print("✅ Scripts de ejecución creados (.bat)")
    return True

def main():
    """Función principal"""
    print("🚀 Configuración del entorno Federated Fog Demo")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Configurar entorno virtual
    if not setup_virtual_environment():
        print("❌ Error configurando entorno virtual")
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        sys.exit(1)
    
    # Verificar instalación
    if not verify_installation():
        print("⚠️  Algunas dependencias podrían no estar instaladas correctamente")
    
    # Probar conexión MQTT
    test_mqtt_connection()
    
    # Crear scripts de ejecución
    create_run_scripts()
    
    print("\n" + "=" * 50)
    print("✅ ¡Configuración completada!")
    print("\n📋 Próximos pasos:")
    
    if platform.system() == "Windows":
        print("   1. Ejecutar los archivos .bat en orden:")
        print("      - run_server.bat")
        print("      - run_fog_broker.bat") 
        print("      - run_fog_client.bat")
        print("      - run_client.bat (uno o más)")
        print("      - run_debug.bat (opcional)")
    else:
        python_exe = get_python_executable()
        print(f"   1. Activar entorno: source .venv/bin/activate")
        print(f"   2. Ejecutar componentes en orden:")
        print(f"      - {python_exe} server.py")
        print(f"      - {python_exe} broker_fog.py")
        print(f"      - {python_exe} fog_flower_client.py")
        print(f"      - {python_exe} client.py")
    
    print("\n📖 Consulta el README.md para más detalles")

if __name__ == "__main__":
    main()
