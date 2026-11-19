"""
ENG499 Cold Chain Logistics — RSL Model Training, Comparison & Error Analysis
Role: CSE-2 (RSL Modeling Lead)

Matched to CSE-1 schema (`rsl_training_table.csv`):
  - Group key: `lot_id`
  - Kinetic baseline feature: `equiv_age_at_4c_days`
  - Target: `target_rsl_days`
  - Engineered features: stage durations, thermal excursions, sensor gaps, `checkpoint`, `store`

Benchmarks:
  - K1 Linear: LinearRegression on `equiv_age_at_4c_days`
  - K2 Kinetic: Fixed-slope kinetic baseline on `equiv_age_at_4c_days`
  - R1 Ridge: Ridge regression on `equiv_age_at_4c_days` + engineered features
  - ML Gradient Boosting: GradientBoostingRegressor + Quantile Uncertainty (P10, P90)

Usage:
  python train_compare.py --data rsl_training_table.csv --output-dir .
  python train_compare.py --demo
"""

import argparse
import json
import os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KINETIC_COL = "equiv_age_at_4c_days"
TARGET_COL = "target_rsl_days"
GROUP_COL = "lot_id"

NUMERIC_FEATURE_CANDIDATES = [
    "equiv_age_at_4c_days",
    "age_h",
    "hours_coldroom",
    "hours_dock",
    "hours_vehicle",
    "hours_vehicle_stop",
    "hours_floor",
    "hours_display",
    "meas_mean_c",
    "meas_max_c",
    "meas_dh_above_4",
    "meas_dh_above_8",
    "meas_missing_h",
    "hours_without_sensor",
    "ambient_max_c_on_route",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "checkpoint",
    "store",
]


class FixedKineticBaseline(BaseEstimator, RegressorMixin):
    """K2 Kinetic baseline: predicts RSL (days) = estimated_initial_shelf_life_days - equiv_age_at_4c_days."""

    def __init__(self, initial_shelf_life_days: float = 14.0):
        self.initial_shelf_life_days = initial_shelf_life_days

    def fit(self, X, y):
        equiv = np.asarray(X, dtype=float)[:, 0]
        self.estimated_initial_ = float(np.mean(np.asarray(y, dtype=float) + equiv))
        return self

    def predict(self, X):
        equiv = np.asarray(X, dtype=float)[:, 0]
        return np.maximum(0.0, self.estimated_initial_ - equiv)


def generate_demo_dataset(n_lots: int = 217, rows_per_lot: int = 55, seed: int = 42) -> pd.DataFrame:
    """Generates a schema-compatible synthetic dataset matching CSE-1 columns."""
    rng = np.random.default_rng(seed)
    checkpoints = ["CP1_PLANT", "CP2_TRANSIT", "CP3_HUB", "CP4_RETAIL"]
    stores = ["STORE_A", "STORE_B", "STORE_C"]
    records = []

    for lot_idx in range(n_lots):
        lot_id = f"LOT_{lot_idx + 1:04d}"
        store = rng.choice(stores)
        base_rsl_days = 14.0 + rng.normal(0, 0.25)
        equiv_days = 0.0
        for step in range(rows_per_lot):
            cp = checkpoints[min(int(step / rows_per_lot * len(checkpoints)), len(checkpoints) - 1)]
            temp_c = rng.normal(4.2 if cp != "CP4_RETAIL" else 5.1, 1.1)
            dt_hours = 4.0
            q10_factor = 2.0 ** ((temp_c - 4.0) / 10.0)
            equiv_days += (dt_hours * q10_factor) / 24.0
            true_rsl_days = max(0.0, base_rsl_days - equiv_days + rng.normal(0, 0.12))
            age_h = step * dt_hours
            records.append(
                {
                    "su_id": f"SU_{lot_idx + 1:04d}_{step:03d}",
                    "lot_id": lot_id,
                    "store": store,
                    "checkpoint": cp,
                    "age_h": age_h,
                    "hours_coldroom": round(age_h * 0.4, 2),
                    "hours_dock": round(age_h * 0.1, 2),
                    "hours_vehicle": round(age_h * 0.25, 2),
                    "hours_vehicle_stop": round(age_h * 0.05, 2),
                    "hours_floor": round(age_h * 0.05, 2),
                    "hours_display": round(age_h * 0.15, 2),
                    "meas_mean_c": round(temp_c, 3),
                    "meas_max_c": round(temp_c + abs(rng.normal(0.5, 0.4)), 3),
                    "meas_dh_above_4": round(max(0.0, temp_c - 4.0) * dt_hours, 3),
                    "meas_dh_above_8": round(max(0.0, temp_c - 8.0) * dt_hours, 3),
                    "meas_missing_h": 0.0,
                    "hours_without_sensor": 0.0,
                    "equiv_age_at_4c_days": round(equiv_days, 5),
                    "ambient_max_c_on_route": round(temp_c + 12.0, 2),
                    "target_rsl_days": round(true_rsl_days, 5),
                }
            )
    return pd.DataFrame(records)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
    }


def run_benchmark(df: pd.DataFrame, output_dir: str = ".") -> dict:
    required_cols = {GROUP_COL, KINETIC_COL, TARGET_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[GROUP_COL, KINETIC_COL, TARGET_COL]).reset_index(drop=True)

    numeric_candidates = [c for c in NUMERIC_FEATURE_CANDIDATES if c in df.columns]
    cat_candidates = [c for c in CATEGORICAL_FEATURE_CANDIDATES if c in df.columns]
    all_features = numeric_candidates + cat_candidates

    # Strict lot-grouped train/test split (80% lots train, 20% lots holdout test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, df[TARGET_COL], groups=df[GROUP_COL]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    y_train = train_df[TARGET_COL].values.astype(float)
    y_test = test_df[TARGET_COL].values.astype(float)
    groups_train = train_df[GROUP_COL].values

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_candidates,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_candidates,
            ),
        ]
    )

    models = {
        "K1_Linear": (
            [KINETIC_COL],
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("reg", LinearRegression())]),
            f"{KINETIC_COL} only (unconstrained linear)",
        ),
        "K2_Kinetic": (
            [KINETIC_COL],
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("reg", FixedKineticBaseline())]),
            f"{KINETIC_COL} only (fixed-slope kinetic law)",
        ),
        "R1_Ridge": (
            all_features,
            Pipeline([("prep", preprocessor), ("reg", Ridge(alpha=1.0))]),
            "Full engineered feature set (L2 regularized)",
        ),
        "ML_Gradient_Boosting": (
            all_features,
            Pipeline(
                [
                    ("prep", preprocessor),
                    (
                        "reg",
                        GradientBoostingRegressor(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.85,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "Full engineered feature set (non-linear ensemble)",
        ),
    }

    gkf = GroupKFold(n_splits=5)
    comparison_rows = []
    results_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "days",
        "target_column": TARGET_COL,
        "kinetic_column": KINETIC_COL,
        "dataset_summary": {
            "total_rows": int(len(df)),
            "total_lots": int(df[GROUP_COL].nunique()),
            "train_rows": int(len(train_df)),
            "train_lots": int(train_df[GROUP_COL].nunique()),
            "test_rows": int(len(test_df)),
            "test_lots": int(test_df[GROUP_COL].nunique()),
            "features_used": all_features,
        },
        "models": {},
    }

    best_gb_pipeline = None
    test_preds_gb = None

    for model_name, (feat_cols, pipeline, desc) in models.items():
        cv_maes, cv_rmses = [], []
        for tr_i, va_i in gkf.split(train_df, y_train, groups=groups_train):
            pipeline.fit(train_df.iloc[tr_i][feat_cols], y_train[tr_i])
            va_pred = np.maximum(0.0, pipeline.predict(train_df.iloc[va_i][feat_cols]))
            m = evaluate_predictions(y_train[va_i], va_pred)
            cv_maes.append(m["mae"])
            cv_rmses.append(m["rmse"])

        pipeline.fit(train_df[feat_cols], y_train)
        test_pred = np.maximum(0.0, pipeline.predict(test_df[feat_cols]))
        holdout_metrics = evaluate_predictions(y_test, test_pred)

        if model_name == "ML_Gradient_Boosting":
            best_gb_pipeline = pipeline
            test_preds_gb = test_pred

        model_record = {
            "description": desc,
            "feature_count": len(feat_cols),
            "holdout_mae_days": holdout_metrics["mae"],
            "holdout_rmse_days": holdout_metrics["rmse"],
            "cv_mae_mean_days": round(float(np.mean(cv_maes)), 4),
            "cv_mae_std_days": round(float(np.std(cv_maes)), 4),
            "cv_rmse_mean_days": round(float(np.mean(cv_rmses)), 4),
            "cv_rmse_std_days": round(float(np.std(cv_rmses)), 4),
        }
        results_summary["models"][model_name] = model_record
        comparison_rows.append({"model": model_name, **model_record})

    # Train P10 and P90 quantile models for M2 uncertainty intervals (in days)
    gb_lower = Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                GradientBoostingRegressor(
                    loss="quantile", alpha=0.10, n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
                ),
            ),
        ]
    )
    gb_upper = Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                GradientBoostingRegressor(
                    loss="quantile", alpha=0.90, n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
                ),
            ),
        ]
    )
    gb_lower.fit(train_df[all_features], y_train)
    gb_upper.fit(train_df[all_features], y_train)

    os.makedirs(output_dir, exist_ok=True)
    artifact = {
        "point_model": best_gb_pipeline,
        "lower_p10_model": gb_lower,
        "upper_p90_model": gb_upper,
        "feature_columns": all_features,
        "numeric_columns": numeric_candidates,
        "categorical_columns": cat_candidates,
        "unit": "days",
    }
    joblib.dump(artifact, os.path.join(output_dir, "rsl_gb_artifact.joblib"))

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # Export stratified M2 error analysis (by checkpoint and store)
    eval_df = test_df.copy()
    eval_df["y_pred"] = test_preds_gb
    eval_df["abs_error"] = np.abs(eval_df[TARGET_COL] - eval_df["y_pred"])
    eval_df["sq_error"] = (eval_df[TARGET_COL] - eval_df["y_pred"]) ** 2

    breakdown_rows = []
    for slice_col in ["checkpoint", "store"]:
        if slice_col in eval_df.columns:
            for val, grp in eval_df.groupby(slice_col):
                breakdown_rows.append(
                    {
                        "slice_dimension": slice_col,
                        "slice_value": str(val),
                        "sample_count": int(len(grp)),
                        "lot_count": int(grp[GROUP_COL].nunique()),
                        "mae_days": round(float(grp["abs_error"].mean()), 4),
                        "rmse_days": round(float(np.sqrt(grp["sq_error"].mean())), 4),
                        "bias_mean_days": round(float((grp[TARGET_COL] - grp["y_pred"]).mean()), 4),
                    }
                )
    pd.DataFrame(breakdown_rows).to_csv(os.path.join(output_dir, "error_analysis_breakdown.csv"), index=False)
    results_summary["error_analysis_slices"] = breakdown_rows

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="ENG499 RSL Model Training & Comparison")
    parser.add_argument("--data", type=str, default=None, help="Path to cleaned CSV dataset (rsl_training_table.csv)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write outputs")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic schema-matching dataset")
    args = parser.parse_args()

    if args.data:
        df = pd.read_csv(args.data)
    elif args.demo:
        df = generate_demo_dataset()
    else:
        raise SystemExit("Please provide --data rsl_training_table.csv or pass --demo.")

    summary = run_benchmark(df, output_dir=args.output_dir)
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
