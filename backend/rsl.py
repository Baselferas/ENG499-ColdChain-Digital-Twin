"""
RSL hook (interface CSE-2 <-> CSE-3).

TODAY  : the twin shows `target_rsl_days` / `label_days_left` / `equiv_age_at_4c_days` of the last checkpoint
         (arrival, display_day_1 ... display_day_5) that has already happened at the selected time.
         This is the *sensor-based reference RSL* of rsl_training_table.csv, not a model prediction.

LATER  : when CSE-2's prediction function is ready, set the environment variable

             RSL_PREDICTOR=rsl_model.predict:predict_rsl

         The function receives ONE argument - a dict with the columns of the checkpoint row of
         rsl_training_table.csv (features) - and returns a dict, e.g.
             {"predicted_days": 5.3, "uncertainty_days": 0.6}
         Nothing else in the backend or dashboard has to change.

Output format used by the API (matches "lot or SU, predicted days, uncertainty"):
    su_id, lot_id, checkpoint, checkpoint_time, predicted_days, uncertainty_days,
    label_days_left, equiv_age_at_4c_days, source
"""
from __future__ import annotations

import importlib
import logging

import pandas as pd

from . import config as cfg

log = logging.getLogger("twin.rsl")

_predictor = None
if cfg.RSL_PREDICTOR:
    try:
        mod_name, fn_name = cfg.RSL_PREDICTOR.split(":")
        _predictor = getattr(importlib.import_module(mod_name), fn_name)
        log.info("Using external RSL predictor %s", cfg.RSL_PREDICTOR)
    except Exception as exc:  # noqa: BLE001 - never crash the backend because of the model
        log.warning("Could not load RSL_PREDICTOR=%s (%s). Falling back to the training-table RSL.", cfg.RSL_PREDICTOR, exc)


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def checkpoint_to_rsl(row: pd.Series) -> dict:
    predicted, uncertainty, source = _num(row.get("target_rsl_days")), None, "rsl_training_table (reference RSL)"
    if _predictor is not None:
        try:
            out = _predictor(row.to_dict())
            predicted = _num(out.get("predicted_days"))
            uncertainty = _num(out.get("uncertainty_days"))
            source = "RSL model (CSE-2)"
        except Exception as exc:  # noqa: BLE001
            log.warning("RSL predictor failed for %s (%s) - using the reference RSL", row.get("su_id"), exc)
    return {
        "su_id": row.get("su_id"),
        "lot_id": row.get("lot_id"),
        "checkpoint": row.get("checkpoint"),
        "checkpoint_time": row.get("checkpoint_time"),
        "predicted_days": predicted,
        "uncertainty_days": uncertainty,
        "label_days_left": _num(row.get("label_days_left")),
        "equiv_age_at_4c_days": _num(row.get("equiv_age_at_4c_days")),
        "source": source,
    }


def rsl_series(ml_su: pd.DataFrame) -> list[dict]:
    """All checkpoints of one SU in time order."""
    return [checkpoint_to_rsl(r) for _, r in ml_su.iterrows()]


def rsl_at(ml_su: pd.DataFrame, as_of) -> dict | None:
    """RSL of the last checkpoint that already happened at `as_of` (None before arrival)."""
    done = ml_su[ml_su["checkpoint_time"] <= pd.Timestamp(as_of)]
    if done.empty:
        return None
    return checkpoint_to_rsl(done.iloc[-1])
