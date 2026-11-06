"""
Digital-twin service layer (CSE-3): turns the raw tables into the "state of a shipment unit".

Main entry points (used by backend/main.py):
    list_units(store, ...)          -> SU list with alert flags
    get_journey(store, su_id)       -> stages, events, outcome   (where it was and when)
    get_temperature(store, su_id)   -> temperature history of the monitored stages
    get_state(store, su_id, as_of)  -> location, last temperature, RSL and alerts at time `as_of`

`as_of` makes the twin replayable: the data is a simulation, so "now" is a point on the SU's journey.
Default = end of the journey (final state).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as cfg
from . import rsl as rsl_mod
from .data_store import DataStore


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def fmt_dur(hours) -> str:
    if hours is None or pd.isna(hours):
        return "n/a"
    minutes = int(round(float(hours) * 60))
    d, rem = divmod(minutes, 1440)
    h, m = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h}h {m:02d}m"


def _py(v):
    """numpy / pandas scalar -> JSON-friendly python value."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(v) else pd.Timestamp(v).isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NaT:
        return None
    return v


# --------------------------------------------------------------------------------------
# journey of one SU
# --------------------------------------------------------------------------------------
class Journey:
    """Everything that is computed once per shipment unit (cached)."""

    def __init__(self, store: DataStore, su_id: str):
        idx = store.index
        rows = idx[idx["su_id"] == su_id]
        if rows.empty:
            raise KeyError(su_id)
        self.row = rows.iloc[0]
        self.su_id = su_id

        ev = store.ev
        prod = ev[(ev["lot_id"] == self.row["lot_id"]) & ev["su_id"].isna() & (ev["event"] == "produced_into_cold_room")]
        own = ev[ev["su_id"] == su_id]
        self.hist = pd.concat([prod, own]).sort_values("timestamp").reset_index(drop=True)
        self.T = self.hist.groupby("event")["timestamp"].min().to_dict()
        if "placed_in_display" not in self.T and pd.notna(self.row["placed_in_display"]):
            self.T["placed_in_display"] = self.row["placed_in_display"]

        if self.T.get("discarded_expired") is not None:
            self.end_time, self.outcome = self.T["discarded_expired"], "Discarded at label expiry"
        elif pd.notna(self.row["last_unit_gone"]):
            self.end_time, self.outcome = self.row["last_unit_gone"], "Sold out"
        else:
            self.end_time, self.outcome = None, "Unknown"

        self.stages = self._build_stages()
        self.temps = self._collect_temperatures(store) if not self.stages.empty else pd.DataFrame()

        ml_su = store.ml[store.ml["su_id"] == su_id].copy()
        ml_su["_ord"] = ml_su["checkpoint"].map({c: i for i, c in enumerate(cfg.CHECKPOINT_ORDER)})
        self.ml = ml_su.sort_values(["_ord", "checkpoint_time"]).drop(columns="_ord").reset_index(drop=True)

        self.lot = store.lots[store.lots["lot_id"] == self.row["lot_id"]]

    # ---------------------------------------------------------------------------------
    @property
    def start(self):
        return self.stages["start"].min() if not self.stages.empty else None

    @property
    def end(self):
        return self.stages["end"].max() if not self.stages.empty else None

    def _build_stages(self) -> pd.DataFrame:
        store_id, veh = self.row["store"], self.row["vehicle_id"]
        T = self.T
        defs = [
            ("1 · Depot cold room", "produced_into_cold_room", "picked_to_dock", "DEPOT_COLDROOM", "Monitored"),
            ("2 · Loading dock", "picked_to_dock", "loaded_on_vehicle", "DOCK", "Blind spot"),
            ("3 · Vehicle transit", "loaded_on_vehicle", "unloading_started", veh, "Monitored"),
            ("4 · Unloading at store", "unloading_started", "received_at_store", veh, "Monitored"),
            ("5 · Store floor", "received_at_store", "placed_in_display", f"{store_id}_FLOOR", "Blind spot"),
            ("6 · Display cabinet", "placed_in_display", None, store_id, "Monitored"),
        ]
        rows = []
        for name, a, b, loc, cov in defs:
            t0 = T.get(a)
            t1 = T.get(b) if b else self.end_time
            if t0 is None or t1 is None or pd.isna(t0) or pd.isna(t1) or t1 < t0:
                continue
            hours = (t1 - t0).total_seconds() / 3600
            rows.append({"stage": name, "location": loc, "start": t0, "end": t1,
                         "duration_h": hours, "duration": fmt_dur(hours), "coverage": cov})
        return pd.DataFrame(rows)

    def _collect_temperatures(self, store: DataStore) -> pd.DataFrame:
        """Readings of the monitored stages only (a parked vehicle is never included)."""
        frames = []
        for _, s in self.stages[self.stages["coverage"] == "Monitored"].iterrows():
            r = store.slice_sensor(s["location"], s["start"], s["end"])
            if r.empty:
                continue
            r["stage"] = s["stage"]
            frames.append(r)
        if not frames:
            return pd.DataFrame()
        t = pd.concat(frames).drop_duplicates(subset=["sensor_id", "timestamp"], keep="first")
        t = t.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

        med = t.groupby("sensor_id")["temperature_c"].transform(
            lambda s: s.rolling(cfg.MEDIAN_WINDOW, center=True, min_periods=1).median()
        )
        t["is_spike"] = (t["temperature_c"] - med).abs() >= cfg.SPIKE_DELTA
        t["temp_used"] = np.where(t["is_spike"], med, t["temperature_c"])
        return t.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)


_CACHE: dict[str, Journey] = {}


def journey(store: DataStore, su_id: str) -> Journey:
    j = _CACHE.get(su_id)
    if j is None:
        if len(_CACHE) > 400:
            _CACHE.clear()
        j = _CACHE[su_id] = Journey(store, su_id)
    return j


# --------------------------------------------------------------------------------------
# API payloads
# --------------------------------------------------------------------------------------
def list_units(store: DataStore, status: str = "all", store_id: str | None = None,
               date_from=None, date_to=None, include_warmup: bool = False) -> list[dict]:
    d = store.index
    if not include_warmup:
        d = d[d["delivery_date"] >= cfg.WARMUP_END]
    if status == "normal":
        d = d[d["normal"]]
    elif status == "alert":
        d = d[d["any_alert"]]
    if store_id:
        d = d[d["store"] == store_id]
    if date_from is not None:
        d = d[d["delivery_date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        d = d[d["delivery_date"] <= pd.Timestamp(date_to)]
    d = d.sort_values(["delivery_date", "su_id"])
    cols = ["su_id", "lot_id", "store", "cabinet_class", "vehicle_id", "route", "crates", "delivery_date",
            "status", "floor_alert", "heat_alert", "red_alert", "floor_h", "arr_target_rsl_days"]
    cols = [c for c in cols if c in d.columns]
    out = []
    for r in d[cols].itertuples(index=False):
        rec = {k: _py(v) for k, v in r._asdict().items()}
        rec["delivery_date"] = str(rec["delivery_date"])[:10]
        rec["rsl_at_arrival_days"] = rec.pop("arr_target_rsl_days", None)
        out.append(rec)
    return out


def get_journey(store: DataStore, su_id: str) -> dict:
    j = journey(store, su_id)
    stages = [{**{k: _py(v) for k, v in s.items()}} for s in j.stages.to_dict("records")]
    ev = j.hist[["timestamp", "event", "location", "qty"]].copy()
    events = [{k: _py(v) for k, v in e.items()} for e in ev.to_dict("records")]
    lot = {}
    if not j.lot.empty:
        lot = {"production_time": _py(pd.Timestamp(j.lot["production_time"].iloc[0])),
               "label_expiry_date": str(j.lot["label_expiry_date"].iloc[0])}
    r = j.row
    return {
        "su_id": su_id, "lot_id": r["lot_id"], "store": r["store"], "cabinet_class": _py(r.get("cabinet_class")),
        "vehicle_id": r["vehicle_id"], "route": _py(r["route"]), "crates": _py(r["crates"]),
        "delivery_date": str(r["delivery_date"])[:10], "outcome": j.outcome, "lot": lot,
        "start": _py(j.start), "end": _py(j.end),
        "blind_spot_hours": float(j.stages.loc[j.stages["coverage"] == "Blind spot", "duration_h"].sum())
        if not j.stages.empty else None,
        "stages": stages, "events": events,
    }


def get_temperature(store: DataStore, su_id: str) -> dict:
    j = journey(store, su_id)
    t = j.temps
    series = []
    if not t.empty:
        for r in t.itertuples(index=False):
            series.append({"timestamp": r.timestamp.isoformat(), "sensor_id": str(r.sensor_id),
                           "stage": r.stage, "temperature_c": round(float(r.temp_used), 2),
                           "raw_c": round(float(r.temperature_c), 2), "is_spike": bool(r.is_spike)})
    return {"su_id": su_id, "alert_threshold_c": cfg.TEMP_ALERT_C, "safe_c": cfg.TEMP_SAFE_C, "series": series}


def get_rsl(store: DataStore, su_id: str, as_of=None) -> dict:
    j = journey(store, su_id)
    series = rsl_mod.rsl_series(j.ml)
    if as_of is not None:
        cur = rsl_mod.rsl_at(j.ml, pd.Timestamp(as_of))
    else:
        cur = series[-1] if series else None
    return {"su_id": su_id, "min_rsl_at_delivery_days": cfg.MIN_RSL_DELIVERY,
            "current": _clean(cur), "checkpoints": [_clean(s) for s in series]}


def _clean(d):
    return None if d is None else {k: _py(v) for k, v in d.items()}


# --------------------------------------------------------------------------------------
# state at time as_of
# --------------------------------------------------------------------------------------
def _stage_at(stages: pd.DataFrame, t: pd.Timestamp):
    """(stage_row or None, phase) where phase in not_started / in_transit_chain / finished."""
    if stages.empty:
        return None, "unknown"
    if t < stages["start"].min():
        return None, "not_started"
    if t >= stages["end"].max():
        return stages.iloc[-1], "finished"
    hit = stages[(stages["start"] <= t) & (t < stages["end"])]
    if hit.empty:                                    # tiny gap between two stages -> previous one
        return stages[stages["start"] <= t].iloc[-1], "active"
    return hit.iloc[0], "active"


def get_state(store: DataStore, su_id: str, as_of=None) -> dict:
    j = journey(store, su_id)
    if j.stages.empty:
        raise ValueError(f"Journey of {su_id} cannot be reconstructed (missing events).")
    t = pd.Timestamp(as_of) if as_of is not None else j.end
    t = min(max(t, j.start), j.end)

    stage, phase = _stage_at(j.stages, t)
    location = {
        "stage": None if stage is None else stage["stage"],
        "asset_id": None if stage is None else stage["location"],
        "monitored": None if stage is None else stage["coverage"] == "Monitored",
        "since": None if stage is None else _py(stage["start"]),
        "time_in_stage": None if stage is None else fmt_dur((t - stage["start"]).total_seconds() / 3600),
        "phase": phase,
        "store": j.row["store"], "vehicle_id": j.row["vehicle_id"],
    }

    # ---- last temperature (<= as_of) --------------------------------------------------
    last_temp, tmax_now, minutes_above = None, None, 0
    tt = j.temps[j.temps["timestamp"] <= t] if not j.temps.empty else j.temps
    if not tt.empty:
        r = tt.iloc[-1]                                   # latest reading (highest if several sensors tie)
        tie = tt[tt["timestamp"] == r["timestamp"]]
        r = tie.sort_values("temp_used").iloc[-1]
        age = (t - r["timestamp"]).total_seconds() / 60
        last_temp = {
            "value_c": round(float(r["temp_used"]), 2), "timestamp": r["timestamp"].isoformat(),
            "sensor_id": str(r["sensor_id"]), "stage": r["stage"], "age_minutes": round(age, 1),
            "stale": bool(age > cfg.STALE_MIN), "spike_corrected": bool(r["is_spike"]),
            "above_limit": bool(r["temp_used"] > cfg.TEMP_ALERT_C),
        }
        tmax_now = float(tt["temp_used"].max())
        # minutes above 8 C (count per sensor, take the sensor with most to avoid double counting)
        over = tt[tt["temp_used"] > cfg.TEMP_ALERT_C]
        minutes_above = int(over.groupby("sensor_id").size().max() * cfg.READING_STEP_MIN) if not over.empty else 0

    # ---- RSL ------------------------------------------------------------------------
    rsl = rsl_mod.rsl_at(j.ml, t)
    rsl_out = _clean(rsl)

    # ---- alerts ---------------------------------------------------------------------
    alerts: list[dict] = []

    def add(code, level, msg):
        alerts.append({"code": code, "level": level, "message": msg})

    if last_temp and last_temp["above_limit"] and not last_temp["stale"]:
        add("TEMP_HIGH_NOW", "critical",
            f"Temperature {last_temp['value_c']:.1f} °C is above {cfg.TEMP_ALERT_C:g} °C ({last_temp['sensor_id']}).")
    elif minutes_above > 0:
        add("TEMP_HIGH_HISTORY", "warning",
            f"Temperature exceeded {cfg.TEMP_ALERT_C:g} °C for about {fmt_dur(minutes_above / 60)} "
            f"(peak {tmax_now:.1f} °C) earlier on this journey.")

    # store-floor wait (ongoing wait counts too)
    rec, plc = j.T.get("received_at_store"), j.T.get("placed_in_display")
    if rec is not None and not pd.isna(rec) and t >= rec:
        floor_end = plc if (plc is not None and not pd.isna(plc) and plc <= t) else t
        floor_h = (floor_end - rec).total_seconds() / 3600
        if floor_h > cfg.FLOOR_WARN_H:
            add("FLOOR_WAIT", "warning", f"Waited {fmt_dur(floor_h)} on the store floor without a sensor "
                                         f"(limit {int(cfg.FLOOR_WARN_H * 60)} min).")

    if rsl is not None and rsl["predicted_days"] is not None and rsl["predicted_days"] < cfg.MIN_RSL_DELIVERY:
        add("RSL_LOW", "critical", f"RSL is {rsl['predicted_days']:.2f} d (< {cfg.MIN_RSL_DELIVERY:g} d minimum).")

    done = j.ml[j.ml["checkpoint_time"] <= t]
    over = done[done["target_over_limit"]]
    if not over.empty:
        add("OVER_LIMIT", "critical", "Spoilage limit exceeded at: " + ", ".join(over["checkpoint"]) + ".")

    level = "critical" if any(a["level"] == "critical" for a in alerts) else ("warning" if alerts else "ok")

    return {
        "su_id": su_id, "lot_id": j.row["lot_id"], "as_of": t.isoformat(),
        "journey_start": _py(j.start), "journey_end": _py(j.end), "outcome": j.outcome,
        "status": level,
        "location": location,
        "last_temperature": last_temp,
        "max_temperature_c": None if tmax_now is None else round(tmax_now, 2),
        "minutes_above_limit": minutes_above,
        "rsl": rsl_out,
        "alerts": alerts,
    }


def latest_by_sensor(store: DataStore) -> list[dict]:
    """Last stored reading of every sensor (used by /sensors/latest when MQTT is off)."""
    out = []
    for asset, g in store.sensors.items():
        for sid, gg in g.groupby("sensor_id"):
            r = gg.iloc[-1]
            out.append({"asset_id": asset, "sensor_id": str(sid), "timestamp": r["timestamp"].isoformat(),
                        "temperature_c": round(float(r["temperature_c"]), 2), "source": "database"})
    return out
