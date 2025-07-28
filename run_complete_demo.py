#!/usr/bin/env python3
"""
Script maestro para ejecutar toda la demo paso a paso
"""

import os
import subprocess
import sys
import time


def run_component(script, name, wait_time=3):
    """Ejecuta un componente y espera"""
    print(f"\n🚀 Iniciando {name}...")
    cmd = f".venv\\Scripts\\python.exe {script}"
    proc = subprocess.Popen(cmd, shell=True)
    print(f"✅ {name} iniciado (PID: {proc.pid})")

    if wait_time > 0:
        print(f"⏳ Esperando {wait_time}s para que se inicialice...")
        time.sleep(wait_time)

    return proc


def main():
    print("🌟 DEMO COMPLETA - Sistema Federated Fog")
    print("=" * 50)

    components = []

    try:
        # 1. Servidor central
        server_proc = run_component("server_fixed.py", "Servidor Flower Central", 4)
        components.append(("Servidor", server_proc))

        # 2. Fog broker
        broker_proc = run_component("broker_fog.py", "Fog Broker", 2)
        components.append(("Fog Broker", broker_proc))

        # 3. Fog client
        fog_client_proc = run_component("fog_flower_client.py", "Fog Flower Client", 3)
        components.append(("Fog Client", fog_client_proc))

        # 4. Clientes locales (3 para hacer agregación)
        for i in range(3):
            client_proc = run_component("client.py", f"Cliente Local {i+1}", 1)
            components.append((f"Cliente {i+1}", client_proc))

        print("\n" + "=" * 50)
        print("✅ TODOS LOS COMPONENTES INICIADOS")
        print("\n📋 Componentes activos:")
        for name, proc in components:
            status = "🟢 Activo" if proc.poll() is None else "🔴 Terminado"
            print(f"   {name}: {status} (PID: {proc.pid})")

        print("\n🎯 El sistema debería estar procesando ahora...")
        print("💡 Presiona Ctrl+C para terminar todos los procesos")

        # Esperar indefinidamente
        while True:
            time.sleep(5)
            # Verificar si algún proceso terminó
            active = 0
            for name, proc in components:
                if proc.poll() is None:
                    active += 1

            if active == 0:
                print("\n⚠️  Todos los procesos han terminado")
                break

    except KeyboardInterrupt:
        print("\n\n🛑 Interrumpido por el usuario")

    finally:
        print("\n🧹 Terminando todos los procesos...")
        for name, proc in components:
            try:
                proc.terminate()
                print(f"   ✅ {name} terminado")
            except BaseException:
                print(f"   ⚠️  {name} ya estaba terminado")

        print("\n✅ Demo finalizada")


if __name__ == "__main__":
    main()
