# broker_fog.py

import json
import threading
import time
from collections import defaultdict

import paho.mqtt.client as mqtt
import numpy as np

# -----------------------------------------------------------------------------
# MQTT TOPICS & CONFIG
# -----------------------------------------------------------------------------
UPDATE_TOPIC   = "fl/updates"        # clients publish Δθ + metadata here
PARTIAL_TOPIC  = "fl/partial"        # this broker publishes region‐partials here
GLOBAL_TOPIC   = "fl/global_model"   # (if you want to re‐publish a full global)

MQTT_BROKER    = "test.mosquitto.org"
MQTT_PORT      = 1883

# Number of per‐region client updates to wait for before computing a partial
K = 3

# -----------------------------------------------------------------------------
# Buffers keyed by region
# -----------------------------------------------------------------------------
buffers = defaultdict(list)


# -----------------------------------------------------------------------------
# Utility: element‐wise weighted average of a list of dicts {param_name: list}
# -----------------------------------------------------------------------------
def weighted_average(updates: list[dict], weights: list[float] = None) -> dict:
    """
    Given a list of N updates (each is a dict of numpy‐serializable lists),
    compute element‐wise average. If `weights` provided, do weighted sum.
    """
    n = len(updates)
    if weights is None:
        weights = [1.0 / n] * n
    avg = {}

    # For each parameter key...
    for key in updates[0]:
        stacked = np.stack([np.array(up[key]) for up in updates], axis=0)
        # weighted sum along axis=0, then convert back to list
        avg[key] = (stacked * np.array(weights)[:, None]).sum(axis=0).tolist()
    return avg


# -----------------------------------------------------------------------------
# MQTT CALLBACK: handle incoming client Δθ
# -----------------------------------------------------------------------------
def on_update(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        region = payload["metadata"]["region"]   # e.g. "us-west"
        weights = payload["weights"]            # dict: param → list

        buffers[region].append(weights)
        print(f"[BROKER] Received update from region={region}. "
              f"Buffer size={len(buffers[region])}/{K}")

        # Once enough updates in this region, compute & publish partial
        if len(buffers[region]) >= K:
            partial = weighted_average(buffers[region])
            buffers[region].clear()

            # Publish the region‐partial for next hop
            msg = {
                "region": region,
                "partial_weights": partial,
                "timestamp": time.time(),
            }
            client.publish(PARTIAL_TOPIC, json.dumps(msg))
            print(f"[BROKER] Published partial for region={region}")

    except Exception as e:
        print("[BROKER ERROR] on_update:", e)


# -----------------------------------------------------------------------------
# THREAD: listen for partials and forward to Flower aggregator
# -----------------------------------------------------------------------------
def start_partial_forwarder():
    """
    In a real system you could call into your Flower server via gRPC here.
    For demo, we simply log or re‐publish to GLOBAL_TOPIC.
    """
    def _on_partial(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            region = data["region"]
            partial = data["partial_weights"]
            print(f"[FORWARDER] Got partial from {region}, size={len(partial)} params")

            # TODO: call Flower aggregator API (gRPC) with `partial`
            # e.g. flower_aggregator.receive_partial(partial)

            # As a placeholder, re‐publish as a “global” update:
            client.publish(GLOBAL_TOPIC, json.dumps(partial))
            print("[FORWARDER] Republished as GLOBAL_MODEL")

        except Exception as e:
            print("[FORWARDER ERROR]", e)

    sub = mqtt.Client()
    sub.on_connect = lambda c, u, f, rc: c.subscribe(PARTIAL_TOPIC)
    sub.on_message = _on_partial
    sub.connect(MQTT_BROKER, MQTT_PORT)
    sub.loop_forever()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # 1) Start forwarder thread
    threading.Thread(target=start_partial_forwarder, daemon=True).start()

    # 2) Start primary broker to collect client updates
    mqttc = mqtt.Client()
    mqttc.on_connect = lambda c, u, f, rc: c.subscribe(UPDATE_TOPIC)
    mqttc.on_message = on_update
    mqttc.connect(MQTT_BROKER, MQTT_PORT)
    print(f"[BROKER] Listening for updates on {UPDATE_TOPIC}")
    mqttc.loop_forever()


if __name__ == "__main__":
    main()
