
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
DATA_DIR = Path("student_data")
TRAIN_PATH = DATA_DIR / "ml" / "rsl_training_table.csv"
LAB_PATH = DATA_DIR / "ml" / "lab_quality_checks.csv"
 
FEATURE = "equiv_age_at_4c_days"
TARGET = "target_rsl_days"
GROUP = "lot_id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
 

LOG_LIMIT = float(np.log10(20000))
GROWTH_RATE_LOG_PER_DAY = None  # e.g. 0.6  <-- set from the project's growth/lag model
LAB_TIME_TOLERANCE = pd.Timedelta("12h")
 
 
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "MAE_days": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_days": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }
 
 
def explore(data):
    print("\n=== DATA EXPLORATION ===")
    print(f"Rows used: {len(data):,}")
    print(f"Unique lots: {data[GROUP].nunique():,}")
    print("\nTarget distribution:")
    print(data[TARGET].describe(percentiles=[.1, .25, .5, .75, .9]).round(3))
    corr = data[FEATURE].corr(data[TARGET])
    print(f"\nCorrelation ({FEATURE} vs {TARGET}): {corr:.3f}")
    print("\nMean RSL by equivalent age (5 quantile groups):")
    try:
        bins = pd.qcut(data[FEATURE], q=5, duplicates="drop")
        print(data.groupby(bins, observed=True)[TARGET].agg(["count", "mean"]).round(3))
    except ValueError as exc:
        print(f"Could not create quantile groups: {exc}")
    return float(corr)
 
 
def make_figures(data, model, test):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figures.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data[TARGET], bins=40, color="#4472C4", edgecolor="white")
    ax.set_xlabel("target_rsl_days")
    ax.set_ylabel("Rows")
    ax.set_title("Distribution of remaining shelf life")
    fig.tight_layout()
    fig.savefig("fig_rsl_distribution.png", dpi=150)
    plt.close(fig)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(test[FEATURE], test[TARGET], s=5, alpha=0.3, label="Test rows")
    xs = np.linspace(data[FEATURE].min(), data[FEATURE].max(), 100)
    ax.plot(xs, model.predict(pd.DataFrame({FEATURE: xs})), color="red", label="Linear fit")
    ax.set_xlabel("equiv_age_at_4c_days")
    ax.set_ylabel("target_rsl_days")
    ax.set_title("RSL vs equivalent age")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_scatter_fit.png", dpi=150)
    plt.close(fig)
 
 
def lab_evaluation(df_full, model):
    """Evaluate against lab plate counts. Returns a dict (never invents RSL labels)."""
    if not LAB_PATH.exists():
        return {"status": "lab file not found"}
    lab = pd.read_csv(LAB_PATH)
    result = {
        "status": "lab file loaded",
        "n_lab_rows": int(len(lab)),
        "lab_columns": list(lab.columns),
        "note": "spc_log_cfu_ml has about +/-0.25 log measurement error.",
    }
    if "spc_log_cfu_ml" not in lab.columns:
        result["status"] = "spc_log_cfu_ml missing"
        return result
 
    # Match lab samples to training rows: same su_id, nearest time within tolerance.
    time_cols = [c for c in df_full.columns
                 if any(k in c.lower() for k in ("time", "date")) or c.lower() in ("ts", "t")]
    if "su_id" not in df_full.columns or "su_id" not in lab.columns or not time_cols \
            or "sample_time" not in lab.columns:
        result["matching"] = ("Cannot match lab samples to training rows: need su_id in both "
                              "tables and a time column in the training table.")
        return result
    tcol = time_cols[0]
    left = lab.copy()
    left["sample_time"] = pd.to_datetime(left["sample_time"], errors="coerce")
    right = df_full.copy()
    right[tcol] = pd.to_datetime(right[tcol], errors="coerce")
    left = left.dropna(subset=["sample_time"]).sort_values("sample_time")
    right = right.dropna(subset=[tcol, FEATURE]).sort_values(tcol)
    m = pd.merge_asof(left, right[["su_id", tcol, FEATURE, TARGET]],
                      left_on="sample_time", right_on=tcol, by="su_id",
                      direction="nearest", tolerance=LAB_TIME_TOLERANCE)
    m = m.dropna(subset=[FEATURE])
    result["n_matched"] = int(len(m))
    if m.empty:
        result["matching"] = "No lab sample matched a training row within tolerance."
        return result
    m["rsl_pred"] = model.predict(m[[FEATURE]])
 
    # (a) Classification check: lab count above the limit <-> model predicts RSL <= 0.
    over = m["spc_log_cfu_ml"] > LOG_LIMIT
    pred_expired = m["rsl_pred"] <= 0
    result["classification"] = {
        "lab_above_limit": int(over.sum()),
        "model_predicts_expired": int(pred_expired.sum()),
        "agreement_rate": float((over == pred_expired).mean()),
    }
 
    # (b) Days-based error, only with a documented growth rate.
    if GROWTH_RATE_LOG_PER_DAY:
        m["rsl_lab"] = ((LOG_LIMIT - m["spc_log_cfu_ml"]) / GROWTH_RATE_LOG_PER_DAY).clip(lower=0)
        result["days_metrics"] = regression_metrics(m["rsl_lab"], m["rsl_pred"])
        result["growth_rate_assumed_log_per_day"] = GROWTH_RATE_LOG_PER_DAY
    else:
        result["days_metrics"] = ("Not computed: set GROWTH_RATE_LOG_PER_DAY from the project's "
                                  "growth/lag model, or obtain matched reference RSL labels.")
    return result
 
 
def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Cannot find {TRAIN_PATH}. Run from the dataset root.")
    df = pd.read_csv(TRAIN_PATH)
    required = {FEATURE, TARGET, GROUP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training table is missing required columns: {sorted(missing)}")
 
    df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[FEATURE, TARGET, GROUP])
    df = df[np.isfinite(df[FEATURE]) & np.isfinite(df[TARGET])].reset_index(drop=True)
    data = df  # keeps extra columns (checkpoint, store, ...) for error analysis
 
    corr = explore(data)
 
    # ---------- 1) Hold-out split by lot ----------
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(data[[FEATURE]], data[TARGET], groups=data[GROUP]))
    train, test = data.iloc[train_idx], data.iloc[test_idx].copy()
    overlap = set(train[GROUP]) & set(test[GROUP])
    if overlap:
        raise RuntimeError(f"Data leakage: {len(overlap)} lot IDs appear in both splits.")
 
    model = LinearRegression().fit(train[[FEATURE]], train[TARGET])
    test["rsl_predicted_days"] = model.predict(test[[FEATURE]])
    test["residual"] = test[TARGET] - test["rsl_predicted_days"]
    holdout = regression_metrics(test[TARGET], test["rsl_predicted_days"])
 
    print("\n=== BASELINE MODEL ===")
    print(f"Formula: predicted_RSL = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * {FEATURE})")
    print(f"Train rows: {len(train):,} ({train[GROUP].nunique()} lots) | "
          f"Test rows: {len(test):,} ({test[GROUP].nunique()} lots)")
    print("\nHold-out test metrics:")
    for k, v in holdout.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
 
    # ---------- 2) GroupKFold ----------
    n_groups = data[GROUP].nunique()
    k = min(N_FOLDS, n_groups)
    folds = []
    for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(data[[FEATURE]], data[TARGET],
                                                              groups=data[GROUP])):
        m = LinearRegression().fit(data.iloc[tr][[FEATURE]], data.iloc[tr][TARGET])
        met = regression_metrics(data.iloc[te][TARGET], m.predict(data.iloc[te][[FEATURE]]))
        met["fold"] = i + 1
        folds.append(met)
    cv = {key: {"mean": float(np.mean([f[key] for f in folds])),
                "std": float(np.std([f[key] for f in folds]))}
          for key in ("MAE_days", "RMSE_days", "R2")}
    print(f"\n{k}-fold GroupKFold (by {GROUP}):")
    for key, v in cv.items():
        print(f"  {key}: {v['mean']:.4f} +/- {v['std']:.4f}")
 
    # ---------- 3) Error analysis by group ----------
    group_rows = []
    for col in ("checkpoint", "store", "store_id", "stage"):
        if col in test.columns:
            g = test.groupby(col).agg(n=("residual", "size"),
                                      mean_residual=("residual", "mean"),
                                      MAE=("residual", lambda r: float(np.abs(r).mean())))
            g.insert(0, "group_column", col)
            group_rows.append(g.reset_index().rename(columns={col: "group_value"}))
    if group_rows:
        pd.concat(group_rows).round(4).to_csv("cse2_residuals_by_group.csv", index=False)
        print("\nSaved cse2_residuals_by_group.csv (positive residual = model under-predicts RSL)")
 
    # ---------- 4) Lab checks ----------
    lab = lab_evaluation(df, model)
    print("\n=== LAB CHECKS ===")
    print(json.dumps(lab, indent=2, default=str))
 
    # ---------- 5) Save ----------
    keep = [c for c in ("su_id", GROUP, "checkpoint", "store") if c in test.columns]
    test[keep + [FEATURE, TARGET, "rsl_predicted_days", "residual"]].to_csv(
        "cse2_test_predictions.csv", index=False)
    results = {
        "model": "LinearRegression",
        "feature": FEATURE,
        "target": TARGET,
        "intercept": float(model.intercept_),
        "coefficient": float(model.coef_[0]),
        "correlation": corr,
        "split": {"method": "GroupShuffleSplit", "group": GROUP, "test_size": TEST_SIZE,
                  "random_state": RANDOM_STATE, "train_rows": int(len(train)),
                  "test_rows": int(len(test)), "train_lots": int(train[GROUP].nunique()),
                  "test_lots": int(test[GROUP].nunique())},
        "holdout_test_metrics": holdout,
        "group_kfold": {"n_splits": k, "folds": folds, "summary": cv},
        "lab_checks": lab,
        "output_field": {"name": "rsl_predicted_days", "unit": "days"},
    }
    with open("cse2_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    make_figures(data, model, test)
    print("\nSaved: cse2_test_predictions.csv, cse2_baseline_results.json, figures.")
 
 
if __name__ == "__main__":
    main()