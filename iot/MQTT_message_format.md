# MQTT message format (CSE-1 -> CSE-3)

This is the format agreed between CSE-1 and CSE-3. The cleaned data keeps long gaps empty and marks them `gap`.

## Topic
```
coldchain/<asset_id>/<sensor_id>/telemetry
```
Examples: `coldchain/U07/TS-U07/telemetry`, `coldchain/DEPOT_COLDROOM/TS-CR-1/telemetry`, `coldchain/V3/TS-V3/telemetry`.
Subscribe to everything with `coldchain/+/+/telemetry`.

## Message (JSON)
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
| `sensor_id`, `asset_id`, `placement` | text | As in `master/sensors.csv` |
| `timestamp` | text | Time of the reading, ISO 8601, local time (`+03:00`). Always the original dataset time |
| `temperature_c` | number or `null` | Clean temperature. `null` only when `quality` is `gap` |
| `raw_temperature_c` | number or `null` | Temperature as measured (before cleaning), `null` if the reading was lost |
| `humidity_pct`, `battery_pct` | number or `null` | `null` only when `quality` is `gap` |
| `quality` | text | `ok`, `spike_removed`, `interpolated` or `gap` |
| `on_board` | `true`, `false` or `null` | Vehicle sensors only: crates are inside. `null` for the other sensors |

### Values of `quality`
- `ok`: measured value, nothing changed
- `spike_removed`: the measured value was a spike, `temperature_c` is an estimate
- `interpolated`: reading lost in a short gap (30 min or less), filled with a straight line
- `gap`: reading lost in a longer gap, `temperature_c` is `null`. The twin must not assume the milk was fine
- (`imputed` exists only if the optional setting `FILL_LONG_GAPS` is switched on in `sensor_cleaning.py`)

## Delivery
- QoS 1. The database ignores a duplicate with the same `sensor_id` and `timestamp`.
- Retain off.
- Broker: Mosquitto on `localhost:1883` for development.
