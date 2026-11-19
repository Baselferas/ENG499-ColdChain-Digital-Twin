# iot – IoT simulation, MQTT stream and incident detection (CSE-1)

ENG499 shelf-life-aware cold chain for milk. This folder is the **IoT / data component**:
it cleans the simulated sensor data, replays it as a live MQTT stream, detects incidents
(refrigeration failure, cabinet failure, door left open, van loaded without pre-cooling,
temperature above 8 °C) and publishes an alert message for each one.

All data is **simulated** (ENG499 Cold Chain Dataset, 25 temperature loggers, one reading every
5 minutes, 1 April – 30 September 2025, 1,317,600 readings).

```
student_data/iot/sensor_readings.csv.gz
        |  sensor_cleaning.py          gaps, spikes, abnormal sensors
        v
output_clean/sensor_readings_clean.csv.gz
        |-- replay_mqtt.py  --> Mosquitto (1883) --> coldchain/<asset>/<sensor>/telemetry --> dashboard / RSL model
        |                                |
        |                                '--> incident_detector.py --live --> coldchain/alerts/<asset>
        |-- incident_detector.py --batch --> detected_incidents.csv  (incident list)
        '-- load_to_db.py / mqtt_to_db.py --> PostgreSQL / TimescaleDB
```

## Files

| File | Purpose |
|---|---|
| `sensor_cleaning.py` | Week 2 cleaning pipeline: gap report, spike removal, short-gap interpolation, abnormal sensors |
| `message_format.py` | Shared code: reads the cleaned file, builds and checks MQTT messages |
| `replay_mqtt.py` | Continuous MQTT stream with adjustable speed (`--day-per-min`, `--speed`, `--loop`) |
| `incident_detector.py` | Incident rules. `--batch` writes the incident list, `--live` publishes alerts |
| `load_to_db.py` | Loads the cleaned file into PostgreSQL / TimescaleDB (safe to run again) |
| `mqtt_to_db.py` | Optional: stores the live telemetry stream in the database |
| `example_subscriber.py` | Example listener for the dashboard developers |
| `db/schema.sql`, `docker-compose.yml`, `mosquitto/` | Database schema and the Docker setup (broker, optional local database) |
| `results/detected_incidents.csv`, `results/incident_summary.md` | Incident list and summary of our run (6,508 incidents) |

The raw data and the cleaned data file are **not** in this repository (see `.gitignore`).
Create the cleaned file with `sensor_cleaning.py`, or get `sensor_readings_clean.csv.gz` from CSE-1.

## Requirements

- Python 3.10 or newer, and `pip install -r requirements.txt` (pandas, numpy, matplotlib, paho-mqtt, psycopg2-binary)
- Docker Desktop (only for the Mosquitto broker, and for the optional local database)

## Quick start (Windows PowerShell, run in this folder)

```powershell
pip install -r requirements.txt

# 1. Clean the raw data (set DATA_DIR in sensor_cleaning.py to your student_data folder)
python sensor_cleaning.py                  # writes output_clean/sensor_readings_clean.csv.gz

# 2. Start the MQTT broker only (Docker Desktop must be running)
docker compose up -d mqtt
docker ps                                  # coldchain-mqtt must be "Up"

# 3. Incident list for the whole period
python incident_detector.py --batch        # writes output_clean/detected_incidents.csv and incident_summary.md

# 4. Live demo: two PowerShell windows
python incident_detector.py --live         # window 1: listens and publishes alerts
python replay_mqtt.py --day-per-min 1 --loop   # window 2: one simulated day per real minute
```

Stop both windows with `Ctrl+C`.

## 1. Continuous MQTT stream (`replay_mqtt.py`)

One message per reading. The `timestamp` in each message is always the original dataset time
(+03:00); only the sending speed changes.

| Option | Meaning |
|---|---|
| `--day-per-min N` | N simulated days per real minute. `1` = 1440 × real time |
| `--speed N` | N times faster than real time (default 1440) |
| `--loop` | start again from the beginning when the data ends |
| `--start`, `--end` | replay a time window, e.g. `--start 2025-07-15 --end 2025-07-16` |
| `--sensors TS-U07 TS-V3` | only some sensors |
| `--fast` | no waiting (tests) |
| `--limit N`, `--dry-run` | print the first N messages without a broker |
| `--host`, `--port` | broker address (default `localhost:1883`) |

At `--day-per-min 1` one 5-minute slot (25 messages, one per sensor) takes 0.208 s, one simulated
day is 7,200 messages (about 120 messages per second), and the whole 183 days take about 3 hours.

### Telemetry message

- **Topic:** `coldchain/<asset_id>/<sensor_id>/telemetry`, for example `coldchain/U07/TS-U07/telemetry`.
  Subscribe to everything with `coldchain/+/+/telemetry`.
- **Delivery:** QoS 1, retain off, Mosquitto on `localhost:1883` (development, no password).

```json
{
  "sensor_id": "TS-U07",
  "asset_id": "U07",
  "placement": "display_cabinet",
  "timestamp": "2025-06-05T10:25:00+03:00",
  "temperature_c": 4.1,
  "raw_temperature_c": 4.4,
  "humidity_pct": 85.0,
  "battery_pct": 71,
  "quality": "ok",
  "on_board": null
}
```

| Field | Meaning |
|---|---|
| `sensor_id`, `asset_id`, `placement` | Sensor, the store / vehicle / cold room it belongs to, and where it is placed |
| `timestamp` | ISO 8601, local time (+03:00), original dataset time |
| `temperature_c` | Clean temperature; `null` only when `quality` is `gap` |
| `raw_temperature_c` | Temperature as measured; `null` if the reading was lost |
| `humidity_pct`, `battery_pct` | `null` only when `quality` is `gap` |
| `quality` | `ok`, `spike_removed`, `interpolated`, `gap` |
| `on_board` | Vehicle sensors only: the crates are inside the vehicle (`true`/`false`), `null` for other sensors |

`gap` means a reading lost in a long gap: show "unknown", never "normal". For vehicles use only
rows with `on_board = true`, because a parked vehicle measures the outside temperature.

## 2. Incident detection (`incident_detector.py`)

One reading is 5 minutes, so 3 readings are 15 minutes. Rows with quality `gap` are skipped.
For vehicles only `on_board = true` readings are used.

| Type | Sensors | Rule |
|---|---|---|
| `temp_above_8c` | all | `temperature_c` > 8.0 °C for at least 15 min. Severity `critical` if the peak is above 10 °C, otherwise `warning` |
| `door_left_open` | display cabinets, cold-room front sensor | Rise of at least 1.5 °C above the median of the previous 30 min and above 7.5 °C (cabinet) / 5.5 °C (cold room), back under 6.5 °C / 5.0 °C after 15 to 90 min |
| `van_loaded_without_precooling` | V1, V2, V3 | When a trip starts (`on_board` becomes true) the first reading is above 8.0 °C and at least 3 of the first 6 readings are above 8.0 °C |
| `cabinet_failure` | display cabinets | Above 9.5 °C for at least 60 min, or above 8.5 °C for at least 120 min |
| `refrigeration_failure` | cold room, vans | Cold room: `TS-CR-1` and `TS-CR-2` both above 7.0 °C for at least 30 min. Vans: more than 30 min after the trip started, above 10.0 °C for at least 30 min |

### Batch mode

```powershell
python incident_detector.py --batch                   # reads output_clean/sensor_readings_clean.csv.gz
python incident_detector.py --batch --publish-batch   # also publishes every incident as an alert
```

Writes `output_clean/detected_incidents.csv` with the columns `asset_id, sensor_id, placement, type,
severity, start, end, duration_min, peak_temp_c, details`, and `output_clean/incident_summary.md`.
`start` and `end` are the first and last reading that fulfils the rule, so `duration_min` is the
number of readings × 5 min.

### Live mode and the alert message

```powershell
python incident_detector.py --live [--host localhost --port 1883]
```

The detector subscribes to `coldchain/+/+/telemetry` and publishes to
**`coldchain/alerts/<asset_id>`** (QoS 1, retain off). Subscribe with `coldchain/alerts/+`.

```json
{
  "alert_id": "ALT-20250718-V2-0001",
  "incident_type": "van_loaded_without_precooling",
  "severity": "critical",
  "asset_id": "V2",
  "sensor_id": "TS-V2",
  "placement": "vehicle_reefer",
  "start_timestamp": "2025-07-18T05:00:00+03:00",
  "end_timestamp": "2025-07-18T05:10:00+03:00",
  "duration_min": 15,
  "peak_temperature_c": 14.2,
  "message": "Vehicle V2 loaded without pre-cooling (started at 14.20 C)"
}
```

(The values are an example of the format.) An alert is sent once, when the rule is first fulfilled,
and is not repeated while the incident continues. For `door_left_open` the alert is sent when the
temperature has recovered, so `start_timestamp`, `end_timestamp` and `duration_min` are the real
values. For the other types `end_timestamp` is the reading that triggered the alert and
`duration_min` is the minimum duration of the rule. There is no message when an incident ends.
The counter in `alert_id` restarts every time the detector is started.

### Results of our run (1 April – 30 September 2025)

| Type | Incidents |
|---|---|
| `temp_above_8c` | 3,226 |
| `door_left_open` | 2,404 |
| `refrigeration_failure` | 631 |
| `van_loaded_without_precooling` | 206 |
| `cabinet_failure` | 41 |
| **Total** | **6,508** |

The 41 cabinet failures (2 to 14 hours, peak about 21 °C) and the cold-room failure on
3 August (315 min, 21 °C) are clear events. Many of the other incidents follow a daily pattern or
the summer heat and are probably normal behaviour, for example the `door_left_open` events at fixed
hours and the vehicle events from June on. See the weeks 5–6 report for the interpretation.

## 3. Database

The cleaned readings are stored in the table `sensor_readings` (`ts`, `sensor_id`, `asset_id`,
`placement`, `temperature_c`, `raw_temperature_c`, `humidity_pct`, `battery_pct`, `quality`,
`on_board`), primary key (`sensor_id`, `ts`). Views: `latest_reading`, `latest_valid_reading`.

The scripts read the connection from options or from environment variables. Do **not** put
passwords in the repository.

```powershell
$env:PGHOST="<database host>"; $env:PGPORT="5432"; $env:PGDATABASE="<database>"
$env:PGUSER="<user>"; $env:PGPASSWORD="<password>"
python load_to_db.py                      # loads the cleaned file; rows that already exist are skipped
python mqtt_to_db.py                      # optional: stores the live stream (start before the replay)
```

Use the host and credentials of the course database (ask CSE-1 or the instructor). Do not use
`--reset` on a shared database: it deletes all readings. As an alternative you can run a local
TimescaleDB with `docker compose up -d` (user, password and database `coldchain`, port 5432; change
the port mapping in `docker-compose.yml` if another PostgreSQL already uses 5432).

