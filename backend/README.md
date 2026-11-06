# backend/ – Digital Twin API (CSE-3)

REST API that returns the **current state of a shipment unit (SU)**: location, last temperature, RSL and alerts.

## Run (from the repository root)
```bash
pip install -r backend/requirements.txt
set COLDCHAIN_DATA=C:\path\to\student_data          # Windows (PowerShell: $env:COLDCHAIN_DATA="...")
python -m uvicorn backend.main:app --port 8000
```
Interactive documentation: http://127.0.0.1:8000/docs

`prepare_sensor_data.py` (agreed data rules) must stay in this folder – the backend imports it.

## Data source for sensor readings
| `SENSOR_SOURCE` | What it reads |
|---|---|
| `csv` (default) | `sensor_readings_clean.csv(.gz)` – found automatically in `COLDCHAIN_DATA`, or set `SENSOR_CSV=<path>` |
| `postgres` | table filled by CSE-1: `PG_DSN`, `PG_TABLE` (default `sensor_readings`), `PG_TEMP_COL` (default `temperature_c`). Needed columns: `timestamp, sensor_id, asset_id, <temp col>, on_board` |

Both paths apply the same rules (vehicle sensors only `on_board == True`, empty temperatures dropped, TS-U07 drift corrected).
Traceability, ML and master tables are read from `COLDCHAIN_DATA` (`traceability/`, `ml/`, `master/`).

## Endpoints
| Method / path | Purpose |
|---|---|
| `GET /shipment-units?status=all\|normal\|alert&store=U07` | SU list with alert flags (`ok` / `warning` / `critical`) |
| `GET /shipment-units/{su_id}/state?as_of=<ISO time>` | **Current state**: location, last temperature, RSL, alerts (default `as_of` = end of the journey) |
| `GET /shipment-units/{su_id}/journey` | Stages (monitored / blind spot) and event log |
| `GET /shipment-units/{su_id}/temperature` | Temperature history of the monitored stages |
| `GET /shipment-units/{su_id}/rsl?as_of=` | RSL checkpoints and the RSL valid at `as_of` |
| `GET /sensors/latest` | Newest reading per sensor (live MQTT values if enabled, otherwise last database row) |
| `GET /stores`, `GET /health` | Helpers |

Timestamps are Gaziantep local time (UTC+03:00) and are returned **without** offset. `as_of` makes the twin replayable
because the data is a simulation.

## Alerts (`/state`)
| Code | Level | Rule |
|---|---|---|
| `TEMP_HIGH_NOW` | critical | last reading > 8 °C (and not stale) |
| `TEMP_HIGH_HISTORY` | warning | readings above 8 °C earlier on the journey (minutes and peak are reported) |
| `FLOOR_WAIT` | warning | store-floor wait > 30 min (an ongoing wait counts) |
| `RSL_LOW` | critical | RSL < 2 days (minimum RSL at delivery) |
| `OVER_LIMIT` | critical | spoilage limit exceeded at a checkpoint that already happened |

## RSL interface (with CSE-2)
Currently the reference RSL of `rsl_training_table.csv` (`target_rsl_days` of the last checkpoint that already happened) is returned.
To plug in CSE-2's model set `RSL_PREDICTOR=rsl_model.predict:predict_rsl`; the function gets the checkpoint row as a dict and returns
`{"predicted_days": ..., "uncertainty_days": ...}`. Output fields: `su_id, lot_id, checkpoint, checkpoint_time, predicted_days,
uncertainty_days, label_days_left, equiv_age_at_4c_days, source`.

## MQTT (optional, with CSE-1)
`MQTT_ENABLED=1 MQTT_HOST=localhost` subscribes to `coldchain/+/+/telemetry` (format agreed with CSE-1) and serves the newest value
of each sensor through `/sensors/latest`. Vehicle messages with `on_board != true` and messages without temperature are ignored.

## Files
`main.py` API · `twin_service.py` state/alert logic · `data_store.py` data loading (CSV / PostgreSQL) · `rsl.py` RSL hook ·
`mqtt_live.py` live feed · `config.py` settings · `prepare_sensor_data.py` agreed data rules
