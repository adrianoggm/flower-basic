# server.py

import json
import torch
import paho.mqtt.client as mqtt
import flwr as fl
from typing import Dict, Optional, Tuple, List

from model import ECGModel
from utils import numpy_to_state_dict, state_dict_to_numpy

# MQTT topics/config
UPDATE_TOPIC  = "fl/partial"       # broker publishes partial aggregates here
MODEL_TOPIC   = "fl/global_model"  # where we publish the new global model
MQTT_BROKER   = "test.mosquitto.org"  # public broker for now

# -----------------------------------------------------------------------------
# 1) Define a custom FedAvg strategy that also implements `evaluate(...)`
# -----------------------------------------------------------------------------
class MQTTFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model: ECGModel,
        mqtt_client: mqtt.Client,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
    ):
        # Pass the correct kwargs into FedAvg
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.global_model = model
        self.mqtt = mqtt_client

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ) -> Optional[fl.common.Parameters]:
        # 1) Perform standard FedAvg aggregation
        new_parameters = super().aggregate_fit(rnd, results, failures)
        if new_parameters is None:
            return None

        # 2) Load aggregated parameters into our local PyTorch model
        state_dict = numpy_to_state_dict(fl.common.parameters_to_ndarrays(new_parameters))
        self.global_model.load_state_dict(state_dict)

        # 3) Re‐serialize and publish the updated global model over MQTT
        payload = state_dict_to_numpy(self.global_model.state_dict())
        self.mqtt.publish(MODEL_TOPIC, json.dumps(payload))

        # 4) Return the new parameters back to Flower
        return new_parameters

    def evaluate(
        self,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """
        Required override of the abstract `evaluate` method.
        Here we simply do no evaluation and return dummy loss=0.0.
        """
        return 0.0, {}

# -----------------------------------------------------------------------------
# 2) Main: start an MQTT listener, instantiate your strategy, and launch FL server
# -----------------------------------------------------------------------------
def main():
    # 2.1) Build your torch model
    model = ECGModel()

    # 2.2) Start MQTT (for receiving partials, publishing globals)
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER)
    mqtt_client.loop_start()

    # 2.3) Instantiate our custom FedAvg strategy
    strategy = MQTTFedAvg(
        model=model,
        mqtt_client=mqtt_client,
        fraction_fit=1.0,          # use all clients each round
        fraction_evaluate=0.0,     # no built‐in eval; we use strategy.evaluate()
        min_fit_clients=2,
        min_evaluate_clients=0,
        min_available_clients=2,
    )

    # 2.4) Launch Flower server with that strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
