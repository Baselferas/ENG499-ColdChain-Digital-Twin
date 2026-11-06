# iot/ – Cleaned sensor data, MQTT stream and time-series database

**Component:** CSE-1, IoT Simulation & Data Lead (ENG499 cold chain project)
**Author:** AbdulGhani Sawas (student no. 240113011052)

This folder turns the raw temperature-logger data into a clean data stream for the rest of
the system. It (1) cleans the readings of all 25 sensors, (2) publishes the cleaned readings
over MQTT, one message per reading, and (3) stores them in a PostgreSQL database (with the TimescaleDB time-series extension, running in Docker)
that the digital-twin backend (CSE-3) reads. All data is **simulated** (ENG499 Cold Chain Dataset).

## How the parts fit together

```
student_data/iot/sensor_readings.csv.gz
        |
        |  sensor_cleaning.py            (gaps, spikes, abnormal sensors, checks)
        v
output_clean/sensor_readings_clean.csv.gz     1,317,600 rows = 25 sensors x 52,704 slots
        |
        |-- load_to_db.py ---------------------------> PostgreSQL/TimescaleDB  (port 5432)
        |
        |-- replay_mqtt.py --> Mosquitto broker (port 1883) --> CSE-3 backend / dashboard
                                        |
                                        '--> mqtt_to_db.py (optional: stores the live stream)
```

## Files

| File | Purpose |
|---|---|
| `sensor_cleaning.py` | Cleaning pipeline. Writes the cleaned file, a summary table, figures and a log to `output_clean/` |
| `docker-compose.yml` | Docker setup: starts the MQTT broker (Mosquitto) and the database (PostgreSQL + TimescaleDB) as two containers |
| `mosquitto/mosquitto.conf` | Broker settings (development: no password) |
| `db/schema.sql` | Tables `sensors`, `sensor_readings` and views `latest_reading`, `latest_valid_reading` |
| `load_to_db.py` | Loads the cleaned file into the database (safe to run again) |
| `replay_mqtt.py` | Publishes the cleaned file over MQTT |
| `mqtt_to_db.py` | Subscriber that stores the MQTT stream in the database (optional) |
| `message_format.py` | Builds and checks MQTT messages (shared code) |
| `example_subscriber.py` | Small example for the backend/dashboard: prints readings, alerts above 8 °C, shows gaps |
| `requirements.txt` | Python packages |

## Requirements
- Python 3.10 or newer
- Docker Desktop (runs the database, PostgreSQL + TimescaleDB, and the Mosquitto broker as containers)
- Python packages: `pip install -r requirements.txt` (pandas, numpy, matplotlib, paho-mqtt, psycopg2-binary)

## 1. Cleaning (already done if `output_clean/` exists)
1. Put the dataset folder `student_data` next to the scripts (or set `DATA_DIR` at the top of `sensor_cleaning.py`).
2. Run `python sensor_cleaning.py`.

What it does, for every sensor on a complete 5-minute grid (1 April – 30 September 2025):
- measures missing readings and the longest gaps;
- removes spikes (more than 5 °C from the rolling median of 7 readings for fixed sensors, 8 °C for vehicle loggers);
- fills gaps of up to 30 minutes by linear interpolation; **longer gaps stay empty and are flagged `gap`** (we cannot know the temperature);
- detects abnormal sensors and their start date (reported, not corrected): **TS-U07** drifts warm from about 1 June 2025 (+2.3 °C by 30 Sep); **TS-U12** was offline from 9 Aug 13:00 to 11 Aug 10:55;
- runs consistency checks; if one fails, the file must not be used.

The `quality` column of the cleaned data:

| quality | Meaning |
|---|---|
| `ok` | measured value, nothing changed |
| `spike_removed` | the measured value was a spike, `temperature_c` is an estimate |
| `interpolated` | reading lost in a short gap (30 min or less), straight-line fill |
| `gap` | reading lost in a longer gap, `temperature_c` is empty (`null`) |
| `imputed` | only if the optional setting `FILL_LONG_GAPS` is on (off by default) |

## 2. Start the broker and the database
```bash
docker compose up -d
docker ps            # coldchain-db and coldchain-mqtt must show "Up"
```
Stop with `docker compose down` (data is kept). `docker compose down -v` also deletes the data.

## 3. Fill the database
```bash
python load_to_db.py            # add --reset to empty the table first
```
Result of our run (full dataset, 36 seconds):

| Check | Result |
|---|---|
| rows in database | 1,317,600 |
| sensors | 25 |
| `ok` | 1,302,041 |
| `interpolated` | 14,326 |
| `gap` | 853 |
| `spike_removed` | 380 |
| non-gap rows without temperature | 0 |

Running the loader again adds nothing (rows with the same `sensor_id` and timestamp are ignored).

## 4. Publish the MQTT stream
```bash
python replay_mqtt.py --start 2025-06-05 --end 2025-06-06   # one day (7,200 messages)
python replay_mqtt.py --fast                                 # everything, no waiting
python replay_mqtt.py --speed 600                            # demo: 1 dataset hour = 6 seconds
python replay_mqtt.py --sensors TS-U07 TS-V3 --fast          # only some sensors
python replay_mqtt.py --dry-run --limit 5                    # print messages, send nothing
```
`--speed N` replays N times faster than real time (default 3000). The `timestamp` in each
message is always the original dataset time; only the sending speed changes.

To watch the stream, run `python example_subscriber.py` in a second window before the replay.
To store the live stream in the database, run `python mqtt_to_db.py` before the replay.

## MQTT message format

**Topic:** `coldchain/<asset_id>/<sensor_id>/telemetry`
Examples: `coldchain/U07/TS-U07/telemetry`, `coldchain/DEPOT_COLDROOM/TS-CR-1/telemetry`, `coldchain/V3/TS-V3/telemetry`.
Subscribe to everything with `coldchain/+/+/telemetry`.

**Delivery:** QoS 1, retain off, broker Mosquitto on `localhost:1883` (development, no password).
The database ignores a duplicate with the same `sensor_id` and `timestamp`.

**Message (JSON), one per reading:**
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

| Field | Type | Meaning |
|---|---|---|
| `sensor_id`, `asset_id`, `placement` | text | Sensor, the store / vehicle / cold room it belongs to, and where it is placed (as in `master/sensors.csv`) |
| `timestamp` | text | Time of the reading, ISO 8601, local time (`+03:00`). Always the original dataset time |
| `temperature_c` | number or `null` | Clean temperature. `null` only when `quality` is `gap` |
| `raw_temperature_c` | number or `null` | Temperature as measured (before cleaning), `null` if the reading was lost |
| `humidity_pct`, `battery_pct` | number or `null` | `null` only when `quality` is `gap` |
| `quality` | text | `ok`, `spike_removed`, `interpolated` or `gap` (see the table in section 1) |
| `on_board` | `true`, `false` or `null` | Vehicle sensors only: crates are inside the vehicle. `null` for the other sensors |

`quality` values: `ok` = measured value, nothing changed; `spike_removed` = the measured value was a
spike, `temperature_c` is an estimate; `interpolated` = reading lost in a short gap (30 min or
less), filled with a straight line; `gap` = reading lost in a longer gap, `temperature_c` is `null`
and the twin must not assume the milk was fine. (`imputed` exists only if the optional setting
`FILL_LONG_GAPS` is switched on in `sensor_cleaning.py`.)

## Database (used by the CSE-3 backend)

| | |
|---|---|
| Host / port | `localhost` (use `127.0.0.1` if you get a password error) / `5432` |
| Database, user, password | `coldchain` / `coldchain` / `coldchain` |
| Main table | `sensor_readings` (`ts`, `sensor_id`, `asset_id`, `placement`, `temperature_c`, `raw_temperature_c`, `humidity_pct`, `battery_pct`, `quality`, `on_board`), primary key (`sensor_id`, `ts`) |
| Views | `latest_reading` (last row per sensor), `latest_valid_reading` (last row with a temperature) |

Rules for using the data:
1. `gap` rows have no temperature. Show "unknown", never "normal".
2. For vehicles use only rows with `on_board = true`; a parked vehicle reads the outdoor temperature.
3. Timestamps are local time (+03:00). Use `SET TIME ZONE 'Europe/Istanbul'` to see them as local time.
4. Take care with TS-U07 after June 2025 (reads too warm) and TS-U12 on 9–11 August (offline).

Example check in the database:
```bash
docker exec -it coldchain-db psql -U coldchain -d coldchain -c "SELECT * FROM latest_valid_reading;"
```

## Troubleshooting
- **`password authentication failed for user "coldchain"`**: another PostgreSQL on the PC may be using port 5432. Try `--host 127.0.0.1`. If that does not help, change `"5432:5432"` to `"5433:5432"` in `docker-compose.yml`, run `docker compose down` and `docker compose up -d`, and add `--host 127.0.0.1 --port 5433` to `load_to_db.py` and `mqtt_to_db.py`.
- **`connection refused`** (database or broker): Docker Desktop is not running, or the containers are not up yet. Check `docker ps`.
- **`Cleaned file not found`**: run `python sensor_cleaning.py` first, or put `sensor_readings_clean.csv.gz` into `output_clean/`.

## Notes and limits
- The raw data and the cleaned file are not stored in the repository (see `.gitignore`); create them with `sensor_cleaning.py`.
- The broker and database have no passwords. They are for development only.
- The vehicle spike threshold (8 °C) does not catch smaller spikes, and a few normal large changes of V3 may be removed.
