"""
Minimal MQTT subscriber (example for CSE-3): prints every message and warns when the
temperature is above 8 C or when a reading is a gap.   python example_subscriber.py
"""
import json

import paho.mqtt.client as mqtt

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="coldchain-example")
except AttributeError:
    client = mqtt.Client(client_id="coldchain-example")


def on_connect(c, userdata, *args):
    c.subscribe("coldchain/+/+/telemetry", qos=1)


def on_message(c, userdata, msg):
    m = json.loads(msg.payload)
    if m["quality"] == "gap":
        print(m["timestamp"], m["sensor_id"], "GAP - no temperature, do not assume the milk was fine")
    elif m["temperature_c"] > 8:
        print(m["timestamp"], m["sensor_id"], "ALERT %.1f C" % m["temperature_c"])
    else:
        print(m["timestamp"], m["sensor_id"], "%.1f C (%s)" % (m["temperature_c"], m["quality"]))


client.on_connect, client.on_message = on_connect, on_message
client.connect("localhost", 1883)
client.loop_forever()
