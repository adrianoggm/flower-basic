"""
client.py - Cliente Local de Aprendizaje Federado

Este módulo implementa clientes locales que:
1. Entrenan modelos CNN en datos ECG5000 locales
2. Publican actualizaciones de modelo vía MQTT al broker fog
3. Reciben modelos globales del servidor central vía MQTT

ARQUITECTURA:
- Cada cliente entrena en su partición local de datos ECG5000
- Envía actualizaciones al broker fog vía topic 'fl/updates'
- El broker fog agrega K clientes por región antes de enviar al servidor central
- Recibe modelos globales actualizados vía topic 'fl/global_model'

FLUJO:
1. Cliente entrena modelo local por 1 época
2. Publica pesos actualizados al broker fog vía MQTT
3. Espera nuevo modelo global del servidor central
4. Actualiza modelo local y repite
"""

import json
import time

import paho.mqtt.client as mqtt
import torch

from .model import ECGModel
from .utils import load_ecg5000_openml

# -----------------------------------------------------------------------------
# CONFIGURACIÓN MQTT
# -----------------------------------------------------------------------------
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_UPDATES = "fl/updates"  # Topic para publicar actualizaciones locales
TOPIC_GLOBAL_MODEL = "fl/global_model"  # Topic para recibir modelos globales


# -----------------------------------------------------------------------------
# CLIENTE LOCAL MQTT
# -----------------------------------------------------------------------------
class FLClientMQTT:
    """
    Cliente de aprendizaje federado que usa MQTT para comunicación.

    Funciones principales:
    1. Entrenamiento local en datos ECG5000 particionados
    2. Publicación de actualizaciones al broker fog
    3. Recepción de modelos globales del servidor central
    4. Sincronización con la arquitectura fog computing
    """

    def __init__(self):
        # Inicializar modelo CNN para ECG
        self.model = ECGModel()

        # Cargar y preparar datos locales ECG5000
        X_train, X_test, y_train, y_test = load_ecg5000_openml()
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            ),
            batch_size=32,
            shuffle=True,
        )

        # Configurar cliente MQTT para comunicación con broker fog
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        print(f"[CLIENT] Conectado a broker MQTT en {MQTT_BROKER}:{MQTT_PORT}")

        # Iniciar loop MQTT en hilo separado
        self.mqtt.loop_start()

        # Flag para detectar cuando llega un nuevo modelo global
        self._got_global = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback MQTT al conectarse: suscribirse a modelos globales."""
        print(
            f"[CLIENT] MQTT conectado (rc={rc}), suscribiéndose a {TOPIC_GLOBAL_MODEL}"
        )
        client.subscribe(TOPIC_GLOBAL_MODEL)

    def _on_message(self, client, userdata, msg):
        """
        Callback MQTT para procesar modelos globales del servidor central.

        Recibe el modelo global actualizado y lo carga en el modelo local
        para la próxima ronda de entrenamiento.
        """
        if msg.topic == TOPIC_GLOBAL_MODEL:
            try:
                payload = json.loads(msg.payload.decode())
                # payload contiene: {"round": X, "global_weights": {param_name:
                # [values]}}
                if "global_weights" in payload:
                    weights_dict = payload["global_weights"]
                    state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
                    self.model.load_state_dict(state_dict, strict=True)
                    round_num = payload.get("round", "?")
                    print(f"[CLIENT] Modelo global cargado de ronda {round_num}")
                    self._got_global = True
                else:
                    print("[CLIENT] Formato de payload inválido recibido")
            except Exception as e:
                print(f"[CLIENT] Error procesando mensaje MQTT: {e}")

    def train_one_round(self):
        """
        Ejecuta una ronda de entrenamiento local y publica la actualización.

        Pasos:
        1. Entrena el modelo local por 1 época en datos ECG5000
        2. Serializa los pesos actualizados
        3. Publica la actualización al broker fog vía MQTT
        4. Espera el próximo modelo global del servidor central
        """
        # Entrenamiento local por 1 época
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()  # Para clasificación binaria ECG
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for X, y in self.train_loader:
            optimizer.zero_grad()
            # Añadir dimensión de canal para CNN: (batch, 1, sequence_length)
            X = X.unsqueeze(1)
            logits = self.model(X).squeeze()  # Eliminar dimensiones extra
            loss = criterion(logits, y.float())  # Convertir a float para BCE
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[CLIENT] Entrenamiento completado. Loss promedio: {avg_loss:.4f}")

        # Serializar pesos actuales y publicar con metadata
        state = self.model.state_dict()
        payload = {
            "client_id": f"client_{id(self) % 1000}",  # ID simple del cliente
            "region": "region_0",  # Región por defecto (puede parametrizarse)
            "weights": {k: v.cpu().numpy().tolist() for k, v in state.items()},
            "num_samples": (
                len(self.train_loader.dataset)
                if hasattr(self.train_loader.dataset, "__len__")
                else len(self.train_loader)
            ),
            "loss": avg_loss,
        }

        self.mqtt.publish(TOPIC_UPDATES, json.dumps(payload))
        print(f"[CLIENT] Actualización local publicada en {TOPIC_UPDATES}")

        # Esperar hasta que el broker fog publique un nuevo modelo global
        print("[CLIENT] Esperando nuevo modelo global...")
        while not self._got_global:
            time.sleep(1)
        self._got_global = False
        print("[CLIENT] Nuevo modelo global recibido, listo para próxima ronda")

    def run(self, rounds: int = 5, delay: float = 5.0):
        """
        Bucle principal del cliente: entrenar→publicar→sincronizar→repetir.

        Args:
            rounds: Número de rondas de entrenamiento federado
            delay: Delay entre rondas (segundos)
        """
        print(f"[CLIENT] Iniciando {rounds} rondas de aprendizaje federado")
        for rnd in range(1, rounds + 1):
            print(f"\n=== Ronda {rnd}/{rounds} ===")
            self.train_one_round()
            if rnd < rounds:  # No hacer delay en la última ronda
                time.sleep(delay)

        # Limpieza de conexiones
        print("[CLIENT] Entrenamiento federado completado, cerrando conexiones")
        self.mqtt.loop_stop()
        self.mqtt.disconnect()


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: Ejecutar cliente local
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Punto de entrada principal para ejecutar un cliente local.

    Configura y ejecuta un cliente que:
    - Entrena en datos ECG5000 locales
    - Participa en aprendizaje federado vía MQTT
    - Se comunica con la arquitectura fog computing
    """
    print("=== CLIENTE LOCAL DE APRENDIZAJE FEDERADO ===")
    print("Iniciando cliente MQTT para fog computing...")

    client = FLClientMQTT()
    client.run(rounds=3, delay=2.0)
