"""
ENG499 Cold Chain Logistics — Production Inference Interface
Role: CSE-2 (RSL Modeling Lead)

Exposed functions for CSE-3 (Backend / Dashboard) and IE-2 / IE-3 (Optimization):
  - predict_rsl(payload: dict, artifact_path: str = ...) -> dict
  - predict_rsl_batch(records: list[dict], artifact_path: str = ...) -> list[dict]
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd

DEFAULT_ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "rsl_gb_artifact.joblib")
KINETIC_COL = "equiv_age_at_4c_days"


@lru_cache(maxsize=2)
def _load_artifact(artifact_path: str) -> Dict[str, Any]:
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{artifact_path}'. "
            "Run `python train_compare.py --data rsl_training_table.csv` first."
        )
    return joblib.load(artifact_path)


def _classify_risk(rsl_lower_days: float) -> str:
    if rsl_lower_days <= 1.0:
        return "CRITICAL"
    if rsl_lower_days <= 3.0:
        return "WARNING"
    return "SAFE"


def predict_rsl(
    shipment_features: Dict[str, Any],
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
) -> Dict[str, Union[float, str, None]]:
    """
    Predicts Remaining Shelf Life (RSL) in days & hours plus M2 uncertainty bounds.

    Args:
        shipment_features: Dictionary containing `equiv_age_at_4c_days` and optional CSE-1 schema columns
            (`age_h`, `hours_coldroom`, `hours_dock`, `hours_vehicle`, `hours_vehicle_stop`,
             `hours_floor`, `hours_display`, `meas_mean_c`, `meas_max_c`, `meas_dh_above_4`,
             `meas_dh_above_8`, `meas_missing_h`, `hours_without_sensor`,
             `ambient_max_c_on_route`, `checkpoint`, `store`, `lot_id`, `su_id`).
        artifact_path: Path to `rsl_gb_artifact.joblib`.
    """
    if KINETIC_COL not in shipment_features or shipment_features[KINETIC_COL] is None:
        raise ValueError(f"Input `shipment_features` must contain a valid numeric `{KINETIC_COL}`.")

    return predict_rsl_batch([shipment_features], artifact_path=artifact_path)[0]


def predict_rsl_batch(
    records: List[Dict[str, Any]],
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
) -> List[Dict[str, Union[float, str, None]]]:
    """Batch inference for multiple lots/checkpoints."""
    if not records:
        return []

    artifact = _load_artifact(artifact_path)
    feature_cols = artifact["feature_columns"]

    aligned_rows = []
    for rec in records:
        if KINETIC_COL not in rec or rec[KINETIC_COL] is None:
            raise ValueError(f"Every record in `records` must include `{KINETIC_COL}`.")
        row = {col: rec.get(col, np.nan) for col in feature_cols}
        aligned_rows.append(row)

    X_df = pd.DataFrame(aligned_rows, columns=feature_cols)

    point_preds = np.maximum(0.0, artifact["point_model"].predict(X_df))
    lower_preds = np.maximum(0.0, artifact["lower_p10_model"].predict(X_df))
    upper_preds = np.maximum(point_preds, artifact["upper_p90_model"].predict(X_df))
    lower_preds = np.minimum(lower_preds, point_preds)

    outputs = []
    for idx, rec in enumerate(records):
        pt_days = round(float(point_preds[idx]), 3)
        lo_days = round(float(lower_preds[idx]), 3)
        hi_days = round(float(upper_preds[idx]), 3)
        outputs.append(
            {
                "su_id": rec.get("su_id"),
                "lot_id": rec.get("lot_id"),
                "rsl_pred_days": pt_days,
                "rsl_pred_hours": round(pt_days * 24.0, 2),
                "rsl_lower_p10_days": lo_days,
                "rsl_upper_p90_days": hi_days,
                "uncertainty_span_days": round(hi_days - lo_days, 3),
                "risk_status": _classify_risk(lo_days),
                "model_version": "ML_Gradient_Boosting_M2",
            }
        )
    return outputs
