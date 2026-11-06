"""
Central configuration of the digital-twin backend (CSE-3).

Everything can be overridden with environment variables, so nobody has to edit code:

    COLDCHAIN_DATA   folder that contains traceability/, ml/, master/ ...   (default: <repo>/student_data)
    SENSOR_SOURCE    csv | postgres                                           (default: csv)
    SENSOR_CSV       path of sensor_readings_clean.csv(.gz)                   (default: auto-detect)
    PG_DSN           e.g. postgresql://user:pass@localhost:5432/coldchain
    PG_TABLE         table filled by CSE-1                                    (default: sensor_readings)
    PG_TEMP_COL      column that holds the cleaned temperature                (default: temperature_c)
    MQTT_ENABLED     1 -> subscribe to coldchain/+/+/telemetry for live values (default: 0)
    MQTT_HOST / MQTT_PORT / MQTT_TOPIC
    RSL_PREDICTOR    optional "module:function" of CSE-2's prediction function
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("COLDCHAIN_DATA", REPO_ROOT / "student_data"))

# ----------------------------- sensor data source ------------------------------------
SENSOR_SOURCE = os.environ.get("SENSOR_SOURCE", "csv").lower()
SENSOR_CSV = os.environ.get("SENSOR_CSV", "")
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/coldchain")
PG_TABLE = os.environ.get("PG_TABLE", "sensor_readings")
PG_TEMP_COL = os.environ.get("PG_TEMP_COL", "temperature_c")

# ----------------------------- MQTT (agreed format with CSE-1) -----------------------
MQTT_ENABLED = os.environ.get("MQTT_ENABLED", "0") == "1"
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "coldchain/+/+/telemetry")

# ----------------------------- RSL ---------------------------------------------------
RSL_PREDICTOR = os.environ.get("RSL_PREDICTOR", "")

# ----------------------------- business rules ----------------------------------------
WARMUP_END = pd.Timestamp("2025-04-01")   # 25-31 Mar 2025 is the warm-up week -> excluded
FLOOR_WARN_H = 0.5                        # store-floor wait threshold (hours)
MIN_RSL_DELIVERY = 2.0                    # stores reject crates with RSL < 2 days
TEMP_ALERT_C = 8.0                        # assignment: alert when temperature > 8 C
TEMP_SAFE_C = 4.0                         # target storage temperature (reference line)
SPIKE_DELTA = 6.0                         # |reading - rolling median| >= 6 C -> sensor spike
MEDIAN_WINDOW = 5
READING_STEP_MIN = 5                      # one reading every 5 minutes
STALE_MIN = 30                            # last reading older than this -> "stale"

CHECKPOINT_ORDER = ["arrival"] + [f"display_day_{i}" for i in range(1, 6)]
