-- Schema of the cold chain time-series database (PostgreSQL, optional TimescaleDB).
-- Safe to run many times. The columns are the same as the fields of the MQTT message.

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'TimescaleDB not available, using plain PostgreSQL';
END $$;

CREATE TABLE IF NOT EXISTS sensors (
    sensor_id  TEXT PRIMARY KEY,
    asset_id   TEXT NOT NULL,          -- store, vehicle (V1..V3) or DEPOT_COLDROOM
    placement  TEXT NOT NULL           -- display_cabinet, vehicle_reefer, cold_room_front, ...
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    ts                 TIMESTAMPTZ      NOT NULL,   -- original dataset time, local time (+03:00)
    sensor_id          TEXT             NOT NULL,
    asset_id           TEXT             NOT NULL,
    placement          TEXT             NOT NULL,
    temperature_c      DOUBLE PRECISION,            -- clean temperature, NULL only if quality = 'gap'
    raw_temperature_c  DOUBLE PRECISION,            -- as measured, NULL if the reading was lost
    humidity_pct       DOUBLE PRECISION,
    battery_pct        SMALLINT,
    quality            TEXT             NOT NULL
        CHECK (quality IN ('ok', 'spike_removed', 'interpolated', 'gap', 'imputed')),
    on_board           BOOLEAN,                     -- vehicle sensors only, NULL for the others
    PRIMARY KEY (sensor_id, ts),                    -- a duplicate message is ignored
    CHECK (quality = 'gap' OR temperature_c IS NOT NULL)
);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('sensor_readings', 'ts',
                                  chunk_time_interval => INTERVAL '30 days',
                                  if_not_exists => TRUE);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_readings_asset_ts ON sensor_readings (asset_id, ts DESC);

-- Last reading of every sensor (also if it is a gap).
CREATE OR REPLACE VIEW latest_reading AS
SELECT DISTINCT ON (sensor_id) *
FROM sensor_readings
ORDER BY sensor_id, ts DESC;

-- Last reading of every sensor that has a temperature (not a gap).
CREATE OR REPLACE VIEW latest_valid_reading AS
SELECT DISTINCT ON (sensor_id) *
FROM sensor_readings
WHERE temperature_c IS NOT NULL
ORDER BY sensor_id, ts DESC;
