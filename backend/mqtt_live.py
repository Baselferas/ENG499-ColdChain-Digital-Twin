"""
Optional live feed (interface CSE-1 <-> CSE-3).

Subscribes to the agreed topic  coldchain/<asset_id>/<sensor_id>/telemetry  (all sensors: coldchain/+/+/telemetry)
and keeps the newest message of every sensor in memory. `GET /sensors/latest` serves it.

Message (agreed with CSE-1), one JSON per reading:
    {"sensor_id": "TS-U07", "asset_id": "U07", "placement": "display_cabinet",
     "timestamp": "2025-06-05T10:25:00+03:00", "temperature_c": 4.1, "raw_temperature_c": 4.1,
     "humidity_pct": 52.0, "battery_pct": 71, "quality": "ok", "on_board": null}

Enable with:  MQTT_ENABLED=1 MQTT_HOST=localhost
The twin never writes to the database from here - CSE-1's loader owns the database.
"""
from __future__ import annotations

import json
import logging
import threading

from . import config as cfg

log = logging.getLogger("twin.mqtt")

LATEST: dict[str, dict] = {}
_lock = threading.Lock()
_client = None
STATUS = {"enabled": cfg.MQTT_ENABLED, "connected": False, "messages": 0}


def _on_connect(client, userdata, *args):
    STATUS["connected"] = True
    client.subscribe(cfg.MQTT_TOPIC)
    log.info("MQTT connected, subscribed to %s", cfg.MQTT_TOPIC)


def _on_disconnect(client, userdata, *args):
    STATUS["connected"] = False


def _on_message(client, userdata, msg):
    try:
        p = json.loads(msg.payload.decode("utf-8"))
        sid = p["sensor_id"]
    except Exception as exc:  # noqa: BLE001
        log.warning("Bad MQTT message on %s: %s", msg.topic, exc)
        return
    # same rule as the batch pipeline: a vehicle reading is only valid while on board
    if str(sid).startswith("TS-V") and p.get("on_board") is not True:
        return
    if p.get("temperature_c") is None:
        return
    with _lock:
        LATEST[sid] = {
            "asset_id": p.get("asset_id"), "sensor_id": sid, "placement": p.get("placement"),
            "timestamp": str(p.get("timestamp"))[:19], "temperature_c": p.get("temperature_c"),
            "quality": p.get("quality"), "battery_pct": p.get("battery_pct"), "source": "mqtt",
        }
    STATUS["messages"] += 1


def start() -> None:
    global _client
    if not cfg.MQTT_ENABLED:
        return
    try:
        import paho.mqtt.client as mqtt

        try:
            _client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)       # paho-mqtt >= 2.0
        except AttributeError:
            _client = mqtt.Client()                                        # paho-mqtt 1.x
        _client.on_connect, _client.on_disconnect, _client.on_message = _on_connect, _on_disconnect, _on_message
        _client.connect_async(cfg.MQTT_HOST, cfg.MQTT_PORT, keepalive=30)
        _client.loop_start()
    except Exception as exc:  # noqa: BLE001 - the API must still start without a broker
        log.warning("MQTT could not be started: %s", exc)


def stop() -> None:
    if _client is not None:
        _client.loop_stop()
        _client.disconnect()


def snapshot() -> list[dict]:
    with _lock:
        return sorted(LATEST.values(), key=lambda r: r["sensor_id"])
