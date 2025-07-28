import json
import time
from typing import List

import numpy as np
import paho.mqtt.client as mqtt
import flwr as fl

from model import ECGModel, get_parameters, set_parameters

# MQTT config
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
PARTIAL_TOPIC = "fl/partial"

# -----------------------------------------------------------------------------
# Flower client running on the fog node
# -----------------------------------------------------------------------------
class FogClient(fl.client.NumPyClient):
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.model = ECGModel()
        self.param_names = list(self.model.state_dict().keys())
        self.partial_weights = None

        # Setup MQTT subscriber to receive partial aggregates
        self.mqtt = mqtt.Client()
        self.mqtt.on_connect = lambda c, u, f, rc: c.subscribe(PARTIAL_TOPIC)
        self.mqtt.on_message = self._on_partial
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT)
        self.mqtt.loop_start()

    # MQTT callback when a partial aggregate is published by broker_fog
    def _on_partial(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            self.partial_weights = data.get("partial_weights")
            print(f"[FOG CLIENT] Received partial for region={data.get('region')}")
        except Exception as e:
            print("[FOG CLIENT] Error processing partial:", e)

    # Flower NumPyClient interface
    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config):
        # Load new global parameters from the central server
        set_parameters(self.model, parameters)

        # Wait for a partial update from local clients
        while self.partial_weights is None:
            time.sleep(0.5)
        # Convert partial dict to parameter list in correct order
        partial_list = [
            np.array(self.partial_weights[name], dtype=np.float32)
            for name in self.param_names
        ]
        self.partial_weights = None
        return partial_list, 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


def main():
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=FogClient("0.0.0.0:8080"))


if __name__ == "__main__":
    main()
