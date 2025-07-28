# debug_mqtt.py
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "#"


def on_connect(client, userdata, flags, rc, properties=None):
    print(
        f"[DEBUG] Connected to MQTT broker {BROKER}:{PORT} (rc={rc}), subscribing to '{TOPIC}'"
    )
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    print(
        f"[DEBUG] {msg.topic} → {msg.payload.decode('utf-8')[:200]}{'...' if len(msg.payload)>200 else ''}"
    )


if __name__ == "__main__":
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    cli.on_connect = on_connect
    cli.on_message = on_message
    cli.connect(BROKER, PORT)
    cli.loop_forever()
