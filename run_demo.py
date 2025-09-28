#!/usr/bin/env python3
"""
Demo completa con mÃºltiples clientes simulando diferentes regiones
"""

import subprocess
import time


def launch_component(
    script_name: str, component_name: str, delay: int = 0
) -> subprocess.Popen:
    """Lanza un componente y espera el delay especificado"""
    if delay > 0:
        print(f"â³ Esperando {delay}s antes de iniciar {component_name}...")
        time.sleep(delay)

    print(f"ð Iniciando {component_name}...")
    cmd = f".venv\\Scripts\\python.exe {script_name}"
    return subprocess.Popen(
        cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def main():
    print("ð DEMO COMPLETA - Federated Learning con Flower")
    print("=" * 60)

    processes = []

    try:
        # 1. Iniciar servidor
        server_proc = launch_component("simple_server.py", "Servidor Flower", 0)
        processes.append(("Servidor", server_proc))

        # 2. Esperar a que el servidor se inicie
        time.sleep(3)

        # 3. Lanzar mÃºltiples clientes en intervalos
        for i in range(3):
            client_proc = launch_component("simple_client.py", f"Cliente {i+1}", 2)
            processes.append((f"Cliente {i+1}", client_proc))

        print("\nâ Todos los componentes iniciados")
        print("ð¡ Observa las ventanas de consola para ver el progreso")
        print("â ï¸  El entrenamiento tomarÃ¡ aproximadamente 30-60 segundos")
        print("\nð Ventanas abiertas:")
        for name, _ in processes:
            print(f"   - {name}")

        print("\nâ+/-ï¸  Esperando que complete el entrenamiento...")

        # Esperar a que terminen los procesos
        time.sleep(45)  # Tiempo estimado para 3 rondas con mÃºltiples clientes

        print("\nð¯ Demo completada")
        print("ð Revisa las ventanas de consola para ver:")
        print("   - Convergencia del modelo")
        print("   - MÃ©tricas de loss por ronda")
        print("   - AgregaciÃ³n de mÃºltiples clientes")

    except KeyboardInterrupt:
        print("\nâ¹ï¸  Demo interrumpida por el usuario")

    finally:
        print("\nð§¹ Limpiando procesos...")
        for name, proc in processes:
            try:
                proc.terminate()
                print(f"   â {name} terminado")
            except BaseException:
                print(f"   â ï¸  {name} ya estaba terminado")


if __name__ == "__main__":
    main()
