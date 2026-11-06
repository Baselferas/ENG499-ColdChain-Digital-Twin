"""CSE-2 Week 2: Kinetic baselines vs Gradient Boosting, same lot_id split, lab checks.

Run from this folder:  python3 train_compare.py <path_to_rsl_training_table.csv> <path_to_lab_quality_checks.csv>
"""
import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from rsl_common import (TARGET, GROUP, KINETIC_FEATURE, FEATURE_GROUPS, ML_FEATURES,
                        LOG_LIMIT, clean_training_table)

TRAIN_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rsl_training_table.csv")
LAB_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("lab_quality_checks.csv")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
SEED, TEST_SIZE, N_FOLDS = 42, 0.20, 5
GB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                 min_samples_leaf=30, random_state=SEED)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": float(r2_score(y, p))}


class FixedSlopeKinetic:
    """RSL = max(0, L - equiv_age_at_4c). One parameter L = total shelf life at 4C (days).
    Physically: 1 equivalent day at 4C consumes exactly 1 day of shelf life."""
    def fit(self, X, y):
        self.L_ = float(np.mean(np.asarray(y) + np.asarray(X).ravel()))
        return self
    def predict(self, X):
        return np.clip(self.L_ - np.asarray(X).ravel(), 0, None)


def make_models():
    return {
        "K1_Linear (equiv_age only)": (lambda: LinearRegression(), [KINETIC_FEATURE]),
        "K2_Kinetic fixed-slope (equiv_age only)": (lambda: FixedSlopeKinetic(), [KINETIC_FEATURE]),
        "R1_Ridge (all features)": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), ML_FEATURES),
        "ML_GradientBoosting (all features)": (lambda: GradientBoostingRegressor(**GB_PARAMS), ML_FEATURES),
    }


def fit_predict(factory, feats, tr, te):
    m = factory().fit(tr[feats], tr[TARGET])
    return m, np.clip(m.predict(te[feats]), 0, None)


def main():
    df = clean_training_table(pd.read_csv(TRAIN_PATH))
    print(f"rows={len(df):,} lots={df[GROUP].nunique()}")

    # ---------- single fair split by lot_id, SAME for all models ----------
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=SEED)
                      .split(df, groups=df[GROUP]))
    train, test = df.iloc[tr_i], df.iloc[te_i].copy()
    assert not set(train[GROUP]) & set(test[GROUP]), "lot leakage"
    print(f"train {len(train):,} rows/{train[GROUP].nunique()} lots | test {len(test):,} rows/{test[GROUP].nunique()} lots")

    models = make_models()
    holdout, fitted = {}, {}
    for name, (fac, feats) in models.items():
        m, p = fit_predict(fac, feats, train, test)
        fitted[name] = m; test[name] = p
        holdout[name] = metrics(test[TARGET], p)

    # ---------- GroupKFold (same folds for every model) ----------
    folds = list(GroupKFold(n_splits=N_FOLDS).split(df, groups=df[GROUP]))
    cv = {}
    for name, (fac, feats) in models.items():
        rows = [metrics(df.iloc[te][TARGET], fit_predict(fac, feats, df.iloc[tr], df.iloc[te])[1])
                for tr, te in folds]
        cv[name] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                        "std": float(np.std([r[k] for r in rows]))} for k in ("MAE", "RMSE", "R2")}

    # ---------- ablation: add feature groups one at a time (GB, hold-out + CV) ----------
    abl, feats = [], [KINETIC_FEATURE]
    steps = [("equiv_age only", [])] + [(f"+ {g}", c) for g, c in FEATURE_GROUPS.items()]
    for label, add in steps:
        feats = feats + add
        _, p = fit_predict(lambda: GradientBoostingRegressor(**GB_PARAMS), feats, train, test)
        cvm = [metrics(df.iloc[te][TARGET],
                       fit_predict(lambda: GradientBoostingRegressor(**GB_PARAMS), feats, df.iloc[tr], df.iloc[te])[1])
               for tr, te in folds]
        h = metrics(test[TARGET], p)
        abl.append({"step": label, "n_features": len(feats), "holdout_MAE": h["MAE"], "holdout_RMSE": h["RMSE"],
                    "cv_MAE": float(np.mean([r["MAE"] for r in cvm])), "cv_RMSE": float(np.mean([r["RMSE"] for r in cvm]))})
    pd.DataFrame(abl).round(4).to_csv(OUT / "ablation_gb.csv", index=False)

    # ---------- error by checkpoint (hold-out) ----------
    gb = "ML_GradientBoosting (all features)"; kb = "K1_Linear (equiv_age only)"
    byc = []
    for c, g in test.groupby("checkpoint"):
        byc.append({"checkpoint": c, "n": len(g),
                    "MAE_K1_linear": mean_absolute_error(g[TARGET], g[kb]),
                    "MAE_GB": mean_absolute_error(g[TARGET], g[gb])})
    pd.DataFrame(byc).round(3).to_csv(OUT / "mae_by_checkpoint.csv", index=False)

    # feature importance (permutation-free: GB impurity importance on train fit)
    imp = pd.Series(fitted[gb].feature_importances_, index=ML_FEATURES).sort_values(ascending=False)
    imp.round(4).to_csv(OUT / "gb_feature_importance.csv", header=["importance"])

    # ---------- uncertainty: out-of-fold (lot-grouped) residual spread of the GB model ----------
    oof = np.zeros(len(df))
    for tr, te in folds:
        oof[te] = np.clip(GradientBoostingRegressor(**GB_PARAMS).fit(df.iloc[tr][ML_FEATURES], df.iloc[tr][TARGET])
                          .predict(df.iloc[te][ML_FEATURES]), 0, None)
    resid = df[TARGET].to_numpy() - oof
    edges = np.quantile(oof, [0, .25, .5, .75, 1.0]); edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(oof, edges[1:-1])
    sigma_bins = [float(np.std(resid[bins == b])) for b in range(4)]
    sigma_global = float(np.std(resid))
    p_te = test[gb].to_numpy(); b_te = np.digitize(p_te, edges[1:-1])
    sig_te = np.array(sigma_bins)[b_te]
    cover = float(np.mean(np.abs(test[TARGET].to_numpy() - p_te) <= 1.2816 * sig_te))
    print(f"OOF sigma global {sigma_global:.3f}; by predicted-RSL quartile {np.round(sigma_bins,3)}; 80% coverage on held-out lots: {cover:.3f}")
    sigma_test = float(sig_te.mean())

    # ---------- final deployable models: fit on ALL rows ----------
    final_gb = GradientBoostingRegressor(**GB_PARAMS).fit(df[ML_FEATURES], df[TARGET])
    final_kin = LinearRegression().fit(df[[KINETIC_FEATURE]], df[TARGET])
    joblib.dump({"gb": final_gb, "kinetic_linear": final_kin,
                 "sigma_edges": edges[1:-1].tolist(), "sigma_bins": sigma_bins, "sigma_global": sigma_global,
                 "features": ML_FEATURES, "kinetic_feature": KINETIC_FEATURE,
                 "trained_rows": int(len(df)), "note": "fit on all rows; honest metrics are from the lot-held-out split"},
                MODELS / "rsl_models.joblib")

    # ---------- lab checks ----------
    lab_res = lab_evaluation(pd.read_csv(TRAIN_PATH), pd.read_csv(LAB_PATH), fitted, lo=None)

    # ---------- save ----------
    test_out = test[["su_id", GROUP, "store", "checkpoint", KINETIC_FEATURE, TARGET] + list(models)].copy()
    test_out.round(4).to_csv(OUT / "test_predictions.csv", index=False)
    comp = pd.DataFrame([{"model": n, "holdout_MAE": holdout[n]["MAE"], "holdout_RMSE": holdout[n]["RMSE"],
                          "holdout_R2": holdout[n]["R2"],
                          "cv_MAE_mean": cv[n]["MAE"]["mean"], "cv_MAE_std": cv[n]["MAE"]["std"],
                          "cv_RMSE_mean": cv[n]["RMSE"]["mean"], "cv_RMSE_std": cv[n]["RMSE"]["std"]}
                         for n in models]).round(4)
    comp.to_csv(OUT / "comparison_mae_rmse.csv", index=False)
    print(comp.to_string(index=False)); print(pd.DataFrame(abl).round(3).to_string(index=False))
    print(pd.DataFrame(byc).round(3).to_string(index=False)); print(imp.round(3).head(8))
    res = {"split": {"method": "GroupShuffleSplit", "group": GROUP, "test_size": TEST_SIZE, "seed": SEED,
                     "train_rows": len(train), "test_rows": len(test),
                     "train_lots": int(train[GROUP].nunique()), "test_lots": int(test[GROUP].nunique())},
           "gb_params": GB_PARAMS, "kinetic_fixed_L_days": fitted["K2_Kinetic fixed-slope (equiv_age only)"].L_,
           "linear_intercept": float(fitted[kb].intercept_), "linear_coef": float(fitted[kb].coef_[0]),
           "holdout": holdout, "group_kfold": cv, "ablation": abl,
           "uncertainty": {"method": "OOF GroupKFold residual SD by predicted-RSL quartile", "sigma_bins": sigma_bins, "sigma_global": sigma_global, "coverage_80pct_on_heldout_lots": cover, "mean_sigma_test": sigma_test},
           "lab": lab_res}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    make_figures(test, models, gb, kb)


def lab_evaluation(full, lab, fitted, lo=None):
    gb_name = "ML_GradientBoosting (all features)"; kb = "K1_Linear (equiv_age only)"
    out = {"n_lab_rows": int(len(lab)), "note": "spc_log_cfu_ml has about +/-0.25 log measurement error"}
    full = full.copy(); full["checkpoint_time"] = pd.to_datetime(full["checkpoint_time"])
    lab = lab.copy(); lab["sample_time"] = pd.to_datetime(lab["sample_time"])
    full = clean_training_table(full.assign()) if False else full
    full = full.dropna(subset=ML_FEATURES).sort_values("checkpoint_time")
    lab = lab.sort_values("sample_time")
    m = pd.merge_asof(lab, full[["su_id", "checkpoint_time", "checkpoint", "target_log_count", TARGET] + ML_FEATURES],
                      left_on="sample_time", right_on="checkpoint_time", by="su_id",
                      direction="nearest", tolerance=pd.Timedelta("12h")).dropna(subset=[KINETIC_FEATURE])
    out["n_matched"] = int(len(m))
    m["dt_h"] = (m["sample_time"] - m["checkpoint_time"]).dt.total_seconds() / 3600
    out["median_abs_time_gap_h"] = float(m["dt_h"].abs().median())
    m["pred_gb"] = np.clip(fitted[gb_name].predict(m[ML_FEATURES]), 0, None)
    m["pred_k1"] = np.clip(fitted[kb].predict(m[[KINETIC_FEATURE]]), 0, None)
    # (1) does the lab count agree with the training table's own bacterial-count column?
    d = m["spc_log_cfu_ml"] - m["target_log_count"]
    out["lab_minus_table_log_count"] = {"mean": float(d.mean()), "sd": float(d.std()),
                                        "within_0.5_log": float((d.abs() <= 0.5).mean())}
    # (2) rank agreement: more remaining life should mean lower bacterial count
    for k in ("pred_gb", "pred_k1"):
        rho, pv = spearmanr(m[k], m["spc_log_cfu_ml"])
        out[f"spearman_{k}_vs_lab_count"] = {"rho": float(rho), "p": float(pv)}
    # (3) classification: lab count above limit vs predicted RSL <= 0 / small
    over = m["spc_log_cfu_ml"] > LOG_LIMIT
    out["classification"] = {"lab_above_limit": int(over.sum()),
                             "gb_pred_rsl_le_1d_among_those": int((m.loc[over, "pred_gb"] <= 1).sum()),
                             "gb_pred_rsl_le_0": int((m["pred_gb"] <= 0).sum()),
                             "mean_pred_rsl_over_limit_samples": float(m.loc[over, "pred_gb"].mean()) if over.any() else None,
                             "mean_pred_rsl_other_samples": float(m.loc[~over, "pred_gb"].mean())}
    # (4) days-based metrics intentionally NOT computed without an agreed growth rate
    out["days_metrics"] = "Not computed: needs a growth rate agreed with the supervisor (lab data are counts, not RSL days)."
    m[["sample_time", "store", "su_id", "lot_id", "checkpoint", "spc_log_cfu_ml", "target_log_count",
       "dt_h", "pred_gb", "pred_k1"]].to_csv(OUT / "lab_matched.csv", index=False, float_format="%.3f")
    return out


def make_figures(test, models, gb, kb):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for a, n, t in zip(ax, [kb, gb], ["Kinetic baseline (linear, equiv_age)", "Gradient Boosting (all features)"]):
        a.scatter(test[TARGET], test[n], s=4, alpha=0.25); lim = [0, 11]
        a.plot(lim, lim, "r--", lw=1); a.set_title(t, fontsize=10); a.set_xlabel("True RSL (days)")
    ax[0].set_ylabel("Predicted RSL (days)"); fig.tight_layout()
    fig.savefig(OUT / "fig_pred_vs_true.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
