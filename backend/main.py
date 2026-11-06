"""
ENG499 Cold-Chain Digital Twin - backend skeleton (CSE-3, Part B)
=================================================================
REST API that returns the current state of a shipment unit (location, last temperature, RSL, alerts).

Run from the repository root:
    pip install -r backend/requirements.txt
    python -m uvicorn backend.main:app --reload --port 8000
Interactive docs:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import config as cfg
from . import mqtt_live
from . import twin_service as svc
from .data_store import DataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

STORE: DataStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STORE
    STORE = DataStore.load()
    mqtt_live.start()
    yield
    mqtt_live.stop()


app = FastAPI(
    title="ENG499 Cold-Chain Digital Twin API",
    version="0.1.0",
    description="Backend skeleton of the digital twin: shipment-unit state, temperature history, RSL and alerts.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _store() -> DataStore:
    if STORE is None:
        raise HTTPException(503, "Data is still loading")
    return STORE


def _parse_time(value: str | None):
    """ISO time (with or without offset). Offsets are dropped: the project uses local time (UTC+03:00)."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"Invalid time '{value}' (use ISO 8601, e.g. 2025-06-05T10:25:00)") from exc
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _check_su(su_id: str) -> None:
    if su_id not in set(_store().index["su_id"]):
        raise HTTPException(404, f"Shipment unit '{su_id}' not found")


# --------------------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    s = _store()
    return {
        "status": "ok",
        "time_zone": "local time of Gaziantep (UTC+03:00); timestamps are returned without offset",
        "sensor_source": s.sensor_source,
        "sensor_rows": s.n_readings,
        "assets": len(s.sensors),
        "shipment_units": int(len(s.index)),
        "mqtt": mqtt_live.STATUS,
        "data_rules": s.prep_report,
    }


@app.get("/stores", tags=["shipment units"])
def stores():
    s = _store()
    return sorted(s.index["store"].dropna().unique().tolist())


@app.get("/shipment-units", tags=["shipment units"])
def shipment_units(
    status: str = Query("all", pattern="^(all|normal|alert)$", description="all | normal | alert"),
    store: str | None = Query(None, description="e.g. U07"),
    date_from: str | None = Query(None, description="delivery date >= (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="delivery date <= (YYYY-MM-DD)"),
    include_warmup: bool = False,
):
    """List of shipment units with alert flags (status: ok / warning / critical)."""
    return svc.list_units(_store(), status, store, _parse_time(date_from), _parse_time(date_to), include_warmup)


@app.get("/shipment-units/{su_id}/state", tags=["shipment units"])
def su_state(su_id: str, as_of: str | None = Query(None, description="ISO time on the SU's journey; default = end")):
    """CURRENT STATE: location, last temperature, RSL and alerts of one shipment unit."""
    _check_su(su_id)
    try:
        return svc.get_state(_store(), su_id, _parse_time(as_of))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/shipment-units/{su_id}/journey", tags=["shipment units"])
def su_journey(su_id: str):
    """Where it was and when: stages (monitored / blind spot) and the event log."""
    _check_su(su_id)
    return svc.get_journey(_store(), su_id)


@app.get("/shipment-units/{su_id}/temperature", tags=["shipment units"])
def su_temperature(su_id: str):
    """Temperature history of all monitored stages (isolated spikes replaced by the rolling median)."""
    _check_su(su_id)
    return svc.get_temperature(_store(), su_id)


@app.get("/shipment-units/{su_id}/rsl", tags=["shipment units"])
def su_rsl(su_id: str, as_of: str | None = None):
    """RSL checkpoints of the SU and the RSL valid at `as_of`."""
    _check_su(su_id)
    return svc.get_rsl(_store(), su_id, _parse_time(as_of))


@app.get("/sensors/latest", tags=["sensors"])
def sensors_latest():
    """Newest reading of every sensor: live MQTT values if MQTT_ENABLED=1, otherwise last row in the database."""
    live = mqtt_live.snapshot()
    return live if live else svc.latest_by_sensor(_store())
