# Federated Fog Demo

This repository shows a minimal prototype of hierarchical aggregation with [Flower](https://flower.ai) and MQTT.
It uses a simple 1D CNN trained on the ECG5000 dataset.

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Optionally download the ECG5000 dataset locally:

```bash
python download_ecg5000.py
```

## Components

- `server.py` – Central Flower server that aggregates partial updates and publishes the global model over MQTT.
- `broker_fog.py` – MQTT broker logic that collects client updates and publishes a partial aggregate after receiving `K` updates from the same region.
- `fog_flower_client.py` – Flower client running on the fog node. It receives partial aggregates from `broker_fog.py` and forwards them to the central server.
- `client.py` – Example local client that trains on ECG5000 and pushes its update to the fog broker via MQTT.

## Running the demo

Start the following processes in separate terminals:

1. **Central server**

   ```bash
   python server.py
   ```

2. **Fog broker** – collects updates from local clients

   ```bash
   python broker_fog.py
   ```

3. **Fog Flower client** – forwards region partials to the central server

   ```bash
   python fog_flower_client.py
   ```

4. **Local clients** – run one or more instances

   ```bash
   python client.py
   ```

Each client trains locally, publishes its update to the broker, and waits for the next global model. The broker computes a partial aggregate and `fog_flower_client.py` sends it to the Flower server, which performs the global aggregation and sends the updated model back over MQTT.
