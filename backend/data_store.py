"""
Data access layer of the digital twin (CSE-3).

Loads, once at start-up:
  * traceability / ML / master tables   (lots, shipment units, events, rsl_training_table, sites)
  * sensor readings                      (CSV written by CSE-1, or the PostgreSQL table CSE-1 fills)

Sensor readings always go through the SAME agreed data rules as the Week-2 pipeline
(backend/prepare_sensor_data.py):
  Rule 1  vehicle sensors: only on_board == True
  Rule 2  rows with empty temperature are dropped
  Rule 3  TS-U07 warm drift (from 1 June) is estimated and subtracted  (original value is never overwritten
          in the source - the corrected value is only used inside the twin)
So the dashboard looks identical whether the readings come from the CSV or from the database.

If CSE-1 chooses InfluxDB instead of PostgreSQL, only `_load_sensors_from_influx()` has to be written
(return the same 5 columns as `_load_sensors_from_postgres()`); nothing else changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import prepare_sensor_data as prep

log = logging.getLogger("twin.data")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def to_naive(series: pd.Series) -> pd.Series:
    """Parse to datetime and drop the timezone (the whole project uses Turkey local time, UTC+03:00)."""
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


# --------------------------------------------------------------------------------------
# sensor sources
# --------------------------------------------------------------------------------------
def _find_sensor_csv() -> Path:
    if cfg.SENSOR_CSV:
        p = Path(cfg.SENSOR_CSV)
        if p.exists():
            return p
        raise FileNotFoundError(f"SENSOR_CSV not found: {p}")
    bases = [cfg.DATA_DIR, cfg.DATA_DIR / "iot", cfg.REPO_ROOT, cfg.REPO_ROOT / "data", Path.cwd()]
    for base in bases:
        for name in prep.DEFAULT_INPUTS:
            if (base / name).exists():
                return base / name
    raise FileNotFoundError(
        "sensor_readings_clean.csv(.gz) not found. Set SENSOR_CSV=<path> or put the file in "
        f"{cfg.DATA_DIR}."
    )


def _read_sensor_csv(path: Path) -> pd.DataFrame:
    """Same as prepare_sensor_data.load_raw, but detects gzip by the file CONTENT, not by the extension
    (a file called .csv may in fact be gzip-compressed)."""
    with open(path, "rb") as fh:
        compression = "gzip" if fh.read(2) == b"\x1f\x8b" else "infer"
    df = pd.read_csv(
        path,
        dtype={"on_board": "string", "sensor_id": "string"},
        parse_dates=["timestamp"],
        compression=compression,
    )
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["on_board"] = df["on_board"].map({"True": True, "False": False}).astype("boolean")
    return df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)


def _load_sensors_from_csv() -> pd.DataFrame:
    path = _find_sensor_csv()
    log.info("Reading sensor readings from %s", path)
    df = _read_sensor_csv(path)
    df["timestamp"] = to_naive(df["timestamp"])

    if "asset_id" not in df.columns:               # cleaned file without asset column -> use master/sensors.csv
        sensors_csv = cfg.DATA_DIR / "master" / "sensors.csv"
        mapping = pd.read_csv(sensors_csv)
        if not {"sensor_id", "asset_id"} <= set(mapping.columns):
            raise RuntimeError("The sensor file has no asset_id column and master/sensors.csv cannot supply it.")
        df = df.merge(mapping[["sensor_id", "asset_id"]].drop_duplicates(), on="sensor_id", how="left")
    df["asset_id"] = df["asset_id"].astype(str)
    return df[["timestamp", "sensor_id", "asset_id", "clean_c", "on_board"]]


def _load_sensors_from_postgres() -> pd.DataFrame:
    """Reads the table that CSE-1's loader fills (columns follow the agreed MQTT message)."""
    from sqlalchemy import create_engine

    log.info("Reading sensor readings from PostgreSQL table %s", cfg.PG_TABLE)
    engine = create_engine(cfg.PG_DSN)
    sql = (
        f'SELECT "timestamp", sensor_id, asset_id, {cfg.PG_TEMP_COL} AS clean_c, on_board '
        f"FROM {cfg.PG_TABLE} ORDER BY sensor_id, \"timestamp\""
    )
    df = pd.read_sql(sql, engine)
    df["timestamp"] = to_naive(df["timestamp"])
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["asset_id"] = df["asset_id"].astype(str)
    df["on_board"] = df["on_board"].map(lambda v: None if v is None or pd.isna(v) else bool(v)).astype("boolean")
    return df


def _load_sensors_from_influx() -> pd.DataFrame:  # pragma: no cover - only if CSE-1 picks InfluxDB
    raise NotImplementedError("Write the Flux query here and return the same columns as the PostgreSQL loader.")


# --------------------------------------------------------------------------------------
# data container
# --------------------------------------------------------------------------------------
@dataclass
class DataStore:
    lots: pd.DataFrame
    su: pd.DataFrame
    ev: pd.DataFrame
    ml: pd.DataFrame
    sites: pd.DataFrame
    sensors: dict                      # asset_id -> DataFrame[timestamp, sensor_id, temperature_c]
    index: pd.DataFrame                # one row per SU with alert flags
    prep_report: dict = field(default_factory=dict)
    sensor_source: str = ""
    n_readings: int = 0

    # ----------------------------------------------------------------------------------
    @classmethod
    def load(cls) -> "DataStore":
        if not cfg.DATA_DIR.exists():
            raise FileNotFoundError(f"Data folder not found: {cfg.DATA_DIR}  (set COLDCHAIN_DATA)")

        lots, su, ev, ml, sites = cls._load_tables()

        if cfg.SENSOR_SOURCE == "postgres":
            raw = _load_sensors_from_postgres()
        elif cfg.SENSOR_SOURCE == "influx":
            raw = _load_sensors_from_influx()
        else:
            raw = _load_sensors_from_csv()

        report: dict = {}
        df = prep.apply_basic_filters(raw, report)          # rules 1 + 2
        df = prep.add_u07_drift_columns(df, report)         # rule 3
        df = df.rename(columns={"clean_c_adj": "temperature_c"})
        df = df.sort_values("timestamp")
        sensors = {
            str(a): g[["timestamp", "sensor_id", "temperature_c"]].reset_index(drop=True)
            for a, g in df.groupby("asset_id")
        }
        log.info("Sensor readings ready: %s rows, %s assets", f"{len(df):,}", len(sensors))

        index = cls._build_index(su, ev, ml, sites)
        return cls(lots, su, ev, ml, sites, sensors, index, report, cfg.SENSOR_SOURCE, len(df))

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _load_tables():
        tr = cfg.DATA_DIR / "traceability"
        lots = pd.read_csv(tr / "lots.csv")

        su = pd.read_csv(tr / "shipment_units.csv")
        for c in ("delivery_date", "placed_in_display", "last_unit_gone"):
            su[c] = pd.to_datetime(su[c], errors="coerce")

        ev = pd.read_csv(tr / "events.csv.gz")
        ev["timestamp"] = to_naive(ev["timestamp"])
        ev = ev.sort_values("timestamp").reset_index(drop=True)

        ml = pd.read_csv(cfg.DATA_DIR / "ml" / "rsl_training_table.csv")
        ml["checkpoint_time"] = to_naive(ml["checkpoint_time"])
        ml["target_over_limit"] = ml["target_over_limit"].map(lambda v: str(v).strip().lower() in ("1", "1.0", "true"))

        sites = pd.read_csv(cfg.DATA_DIR / "master" / "sites.csv")
        return lots, su, ev, ml, sites

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _build_index(su, ev, ml, sites) -> pd.DataFrame:
        """One row per SU with the alert flags used by the SU list (same logic as the Week-2 screen)."""
        ex = ev[ev["su_id"].notna() & ev["event"].isin(["received_at_store", "placed_in_display"])]
        ts = ex.groupby(["su_id", "event"])["timestamp"].min().unstack()
        floor_h = ((ts["placed_in_display"] - ts["received_at_store"]).dt.total_seconds() / 3600).rename("floor_h")

        arrival = ml[ml["checkpoint"] == "arrival"].set_index("su_id")[["target_rsl_days"]].add_prefix("arr_")
        arrival = arrival[~arrival.index.duplicated(keep="first")]
        agg = ml.groupby("su_id").agg(max_dh8=("meas_dh_above_8", "max"), any_over=("target_over_limit", "max"))

        sites_small = sites[["site_id", "cabinet_class"]].set_index("site_id")
        idx = su.set_index("su_id").join(sites_small, on="store").join(floor_h).join(arrival).join(agg)

        idx["floor_alert"] = (idx["floor_h"] > cfg.FLOOR_WARN_H).fillna(False).astype(bool)
        idx["heat_alert"] = (idx["max_dh8"].fillna(0) > 0).astype(bool)
        idx["red_alert"] = ((idx["arr_target_rsl_days"] < cfg.MIN_RSL_DELIVERY).fillna(False)
                            | idx["any_over"].fillna(False).astype(bool)).astype(bool)
        idx["any_alert"] = idx["floor_alert"] | idx["heat_alert"] | idx["red_alert"]
        idx["normal"] = idx["arr_target_rsl_days"].notna() & ~idx["any_alert"]
        idx["status"] = np.where(idx["red_alert"], "critical", np.where(idx["any_alert"], "warning", "ok"))
        return idx.reset_index()

    # ----------------------------------------------------------------------------------
    def slice_sensor(self, asset: str, t0, t1) -> pd.DataFrame:
        """Readings of one asset in [t0, t1]."""
        g = self.sensors.get(str(asset))
        if g is None:
            return pd.DataFrame()
        lo = g["timestamp"].searchsorted(pd.Timestamp(t0), side="left")
        hi = g["timestamp"].searchsorted(pd.Timestamp(t1), side="right")
        return g.iloc[lo:hi].copy()
