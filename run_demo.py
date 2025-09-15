#!/usr/bin/env python3
"""
Demo completa con múltiples clientes simulando diferentes regiones
"""

import subprocess
import time


def launch_component(
    script_name: str, component_name: str, delay: int = 0
) -> subprocess.Popen:
    """Lanza un componente y espera el delay especificado"""
    if delay > 0:
        print(f"⏳ Esperando {delay}s antes de iniciar {component_name}...")
        time.sleep(delay)

    print(f"🚀 Iniciando {component_name}...")
    cmd = f".venv\\Scripts\\python.exe {script_name}"
    return subprocess.Popen(
        cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def main():
    print("🌟 DEMO COMPLETA - Federated Learning con Flower")
    print("=" * 60)

    processes = []

    try:
        # 1. Iniciar servidor
        server_proc = launch_component("simple_server.py", "Servidor Flower", 0)
        processes.append(("Servidor", server_proc))

        # 2. Esperar a que el servidor se inicie
        time.sleep(3)

        # 3. Lanzar múltiples clientes en intervalos
        for i in range(3):
            client_proc = launch_component("simple_client.py", f"Cliente {i+1}", 2)
            processes.append((f"Cliente {i+1}", client_proc))

        print("\n✅ Todos los componentes iniciados")
        print("💡 Observa las ventanas de consola para ver el progreso")
        print("⚠️  El entrenamiento tomará aproximadamente 30-60 segundos")
        print("\n📋 Ventanas abiertas:")
        for name, _ in processes:
            print(f"   - {name}")

        print("\n⏱️  Esperando que complete el entrenamiento...")

        # Esperar a que terminen los procesos
        time.sleep(45)  # Tiempo estimado para 3 rondas con múltiples clientes

        print("\n🎯 Demo completada")
        print("📊 Revisa las ventanas de consola para ver:")
        print("   - Convergencia del modelo")
        print("   - Métricas de loss por ronda")
        print("   - Agregación de múltiples clientes")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrumpida por el usuario")

    finally:
        print("\n🧹 Limpiando procesos...")
        for name, proc in processes:
            try:
                proc.terminate()
                print(f"   ✅ {name} terminado")
            except BaseException:
                print(f"   ⚠️  {name} ya estaba terminado")


if __name__ == "__main__":
    main()
