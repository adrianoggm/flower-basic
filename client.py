# client_broker.py

import time
import json
import torch
from model import ECGModel
from utils import load_data  # your data loader
import paho.mqtt.client as mqtt

# -----------------------------------------------------------------------------
# MQTT CONFIG
# -----------------------------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT   = 1883
TOPIC_UPDATES     = "fl/updates"       # where we publish our local Δθ
TOPIC_GLOBAL_MODEL = "fl/global_model" # where we receive the aggregated θ

# -----------------------------------------------------------------------------
# CLIENT
# -----------------------------------------------------------------------------
class FLClientMQTT:
    def __init__(self):
        # 1) Initialize model
        self.model = ECGModel()
        # 2) Load data into DataLoader(s)
        X_train, X_test, y_train, y_test = load_data()
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            ),
            batch_size=32,
            shuffle=True,
        )

        # 3) Setup MQTT client
        self.mqtt = mqtt.Client()
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

        # Start network loop in background thread
        self.mqtt.loop_start()

        # Flag to know when a new global model has arrived
        self._got_global = False

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Connected (rc={rc}), subscribing to {TOPIC_GLOBAL_MODEL}")
        client.subscribe(TOPIC_GLOBAL_MODEL)

    def _on_message(self, client, userdata, msg):
        if msg.topic == TOPIC_GLOBAL_MODEL:
            payload = json.loads(msg.payload.decode())
            # payload is dict: key→list
            state_dict = {
                k: torch.tensor(v) for k, v in payload.items()
            }
            self.model.load_state_dict(state_dict, strict=True)
            print("[MQTT] Received & loaded new global model")
            self._got_global = True

    def train_one_round(self):
        """Run one local epoch, compute Δθ, publish it, then wait for next global."""
        # 1) Train one epoch
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        for X, y in self.train_loader:
            optimizer.zero_grad()
            logits = self.model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # 2) Compute current θ to publish as 'delta'
        state = self.model.state_dict()
        payload = {k: v.cpu().numpy().tolist() for k, v in state.items()}
        self.mqtt.publish(TOPIC_UPDATES, json.dumps(payload))
        print("[MQTT] Published local Δθ to topic", TOPIC_UPDATES)

        # 3) Wait until fog broker publishes a new global model
        print("[CLIENT] Waiting for new global model...")
        while not self._got_global:
            time.sleep(1)
        self._got_global = False

    def run(self, rounds: int = 5, delay: float = 5.0):
        """Main loop: train→publish→sync→repeat."""
        for rnd in range(1, rounds + 1):
            print(f"\n=== Round {rnd} ===")
            self.train_one_round()
            time.sleep(delay)

        # Clean up
        self.mqtt.loop_stop()
        self.mqtt.disconnect()
        print("Done training.")

if __name__ == "__main__":
    client = FLClientMQTT()
    client.run(rounds=3, delay=2.0)
