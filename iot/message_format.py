"""
Shared helpers: read the cleaned file and build MQTT messages in the agreed format
(see MQTT_message_format.md). Used by replay_mqtt.py, load_to_db.py and mqtt_to_db.py.
"""
import json
import os

import numpy as np
import pandas as pd

CLEAN_FILE = os.path.join("output_clean", "sensor_readings_clean.csv.gz")
TZ_SUFFIX = "+03:00"        # the dataset uses local time; Turkey is UTC+3 all year
QUALITY_VALUES = {"ok", "spike_removed", "interpolated", "gap", "imputed"}
MESSAGE_FIELDS = ["sensor_id", "asset_id", "placement", "timestamp", "temperature_c",
                  "raw_temperature_c", "humidity_pct", "battery_pct", "quality", "on_board"]


def topic_for(asset_id, sensor_id):
    return "coldchain/%s/%s/telemetry" % (asset_id, sensor_id)


def read_clean(path=CLEAN_FILE, chunksize=None):
    """Read the cleaned file. Columns are returned in the message format."""
    cols = ["timestamp", "sensor_id", "asset_id", "placement", "temperature_c",
            "raw_temperature_c", "humidity_pct", "battery_pct", "quality", "on_board"]
    return pd.read_csv(path, usecols=cols, dtype={"on_board": "object"}, chunksize=chunksize)


def _bool_or_none(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0"):
        return True
    if s in ("false", "0", "0.0"):
        return False
    return None


def _num(v, ndigits=None):
    if v is None or pd.isna(v):
        return None
    return round(float(v), ndigits) if ndigits is not None else float(v)


def prepare(df):
    """Add an ISO 8601 timestamp with +03:00 and sort by time, then sensor."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S") + TZ_SUFFIX
    df["dtime"] = ts
    return df.sort_values(["dtime", "sensor_id"]).reset_index(drop=True)


def row_to_message(r):
    """r: a row (namedtuple or Series) -> dict in the agreed message format."""
    bat = None if pd.isna(r.battery_pct) else int(round(float(r.battery_pct)))
    return {
        "sensor_id": r.sensor_id,
        "asset_id": r.asset_id,
        "placement": r.placement,
        "timestamp": r.timestamp,
        "temperature_c": _num(r.temperature_c, 2),
        "raw_temperature_c": _num(r.raw_temperature_c, 2),
        "humidity_pct": _num(r.humidity_pct, 2),
        "battery_pct": bat,
        "quality": r.quality,
        "on_board": _bool_or_none(r.on_board),
    }


def to_json(msg):
    return json.dumps(msg, separators=(",", ":"))


def check_message(m):
    """Return a list of problems (empty list = message follows the agreed format)."""
    problems = []
    if list(m.keys()) != MESSAGE_FIELDS:
        problems.append("fields differ from the agreed format")
    if m["quality"] not in QUALITY_VALUES:
        problems.append("unknown quality %r" % m["quality"])
    if m["quality"] == "gap":
        for k in ("temperature_c", "humidity_pct", "battery_pct"):
            if m[k] is not None:
                problems.append("%s must be null in a gap" % k)
    elif m["temperature_c"] is None:
        problems.append("temperature_c is null but quality is %s" % m["quality"])
    if not str(m["timestamp"]).endswith(TZ_SUFFIX):
        problems.append("timestamp has no +03:00")
    return problems
