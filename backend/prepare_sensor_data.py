"""
ENG499 Cold-Chain Digital Twin - sensor data preparation (CSE-3 / CSE-2 stages)
===============================================================================
Reads sensor_readings_clean.csv (or the .gz version) directly - no PostgreSQL needed -
and applies the agreed data rules:

  Rule 1  Vehicle sensors (TS-V1..TS-V3): keep ONLY rows with on_board == True.
          (Parked rows reach ~46 C because they measure the outside air.)
          NOTE: on_board is empty (NaN) for every NON-vehicle sensor, so the filter
          is applied to vehicle sensors only. A global `df[df.on_board == True]`
          would silently delete all depot / store / display sensors.
  Rule 2  Drop rows where clean_c is empty (NaN).
  Rule 3  TS-U07 reads too warm from ~1 June onward and the file has no label for it.
          The original clean_c is NEVER overwritten. Instead the script adds:
            u07_drift_suspect  True for TS-U07 rows from 2025-06-01
            u07_drift_est_c    estimated warm bias (C), estimated against the other
                               store sensors (robust Theil-Sen trend)
            clean_c_adj        clean_c minus the estimated bias for TS-U07,
                               identical to clean_c for every other sensor
  Rule 4  Extreme temperatures are KEPT. Only diagnostic columns are added:
            is_extreme_cold      clean_c < 0 C
            is_extreme_hot       clean_c > 12 C (non-vehicle sensors)
            suspect_isolated_dip |clean_c - rolling median(5)| >= 6 C
                                 (same rule as dashboard/app.py), so you can see
                                 which extremes are single readings and which are
                                 sustained events.

Usage:
    python prepare_sensor_data.py
    python prepare_sensor_data.py --input sensor_readings_clean.csv --output sensor_readings_prepared.csv

Import from other code (e.g. the Streamlit dashboard):
    from prepare_sensor_data import prepare
    df, report = prepare("sensor_readings_clean.csv")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
VEHICLE_SENSORS = ["TS-V1", "TS-V2", "TS-V3"]
STORE_SENSORS = [f"TS-U{i:02d}" for i in range(1, 16)]
U07 = "TS-U07"
U07_DRIFT_START = pd.Timestamp("2025-06-01")

COLD_LIMIT_C = 0.0          # clean_c below this  -> is_extreme_cold
HOT_LIMIT_C = 12.0          # clean_c above this  -> is_extreme_hot (non-vehicle only)
SPIKE_DELTA = 6.0           # same values as dashboard/app.py
MEDIAN_WINDOW = 5
STEP = pd.Timedelta(minutes=5)

DEFAULT_INPUTS = ["sensor_readings_clean.csv", "sensor_readings_clean_csv.gz", "sensor_readings_clean.csv.gz"]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the CSV / CSV.GZ with correct dtypes (on_board is mixed True/False/empty)."""
    df = pd.read_csv(
        path,
        dtype={"on_board": "string", "sensor_id": "string"},
        parse_dates=["timestamp"],
    )
    df["sensor_id"] = df["sensor_id"].astype(str)
    df["on_board"] = df["on_board"].map({"True": True, "False": False}).astype("boolean")
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------
# Rules 1 and 2
# --------------------------------------------------------------------------------------
def apply_basic_filters(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    is_vehicle = df["sensor_id"].isin(VEHICLE_SENSORS)

    # Rule 1: vehicles -> only on_board == True. Non-vehicle sensors are untouched.
    keep_vehicle_rule = ~is_vehicle | df["on_board"].fillna(False).astype(bool)
    report["rows_in"] = len(df)
    report["rule1_vehicle_rows_total"] = int(is_vehicle.sum())
    report["rule1_vehicle_rows_on_board"] = int((is_vehicle & keep_vehicle_rule).sum())
    report["rule1_rows_removed"] = int((~keep_vehicle_rule).sum())
    df = df.loc[keep_vehicle_rule].copy()

    # Rule 2: clean_c empty -> drop
    n_nan = int(df["clean_c"].isna().sum())
    report["rule2_rows_removed"] = n_nan
    report["rule2_nan_by_sensor"] = df.loc[df["clean_c"].isna(), "sensor_id"].value_counts().to_dict()
    df = df.dropna(subset=["clean_c"]).copy()

    report["rows_out"] = len(df)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Rule 3: TS-U07 drift (flag + estimate, original values untouched)
# --------------------------------------------------------------------------------------
def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust line fit: median of pairwise slopes (insensitive to hot events)."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j][ok] - y[i][ok]) / dx[ok]))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def add_u07_drift_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    df["u07_drift_suspect"] = (df["sensor_id"] == U07) & (df["timestamp"] >= U07_DRIFT_START)
    df["u07_drift_est_c"] = 0.0
    df["clean_c_adj"] = df["clean_c"]

    stores = df[df["sensor_id"].isin(STORE_SENSORS)].copy()
    stores["day"] = stores["timestamp"].dt.floor("D")
    daily = stores.pivot_table(index="day", columns="sensor_id", values="clean_c", aggfunc="median")
    if U07 not in daily.columns:
        report["u07"] = "TS-U07 not found - skipped"
        return df

    peers = daily.drop(columns=U07).median(axis=1)
    diff = (daily[U07] - peers).dropna()

    pre = diff[diff.index < U07_DRIFT_START]
    post = diff[diff.index >= U07_DRIFT_START]
    baseline = float(pre.median())                       # U07's normal offset to its peers
    excess = post - baseline

    x = (excess.index - U07_DRIFT_START).days.to_numpy(dtype=float)
    slope, intercept = _theil_sen(x, excess.to_numpy(dtype=float))

    mask = df["u07_drift_suspect"]
    days = (df.loc[mask, "timestamp"] - U07_DRIFT_START) / pd.Timedelta(days=1)
    est = np.clip(intercept + slope * days, 0.0, None)
    df.loc[mask, "u07_drift_est_c"] = est.round(3)
    df.loc[mask, "clean_c_adj"] = (df.loc[mask, "clean_c"] - est).round(3)

    end_day = (df["timestamp"].max() - U07_DRIFT_START) / pd.Timedelta(days=1)
    report["u07"] = {
        "baseline_offset_vs_peers_c": round(baseline, 2),
        "drift_slope_c_per_day": round(slope, 4),
        "drift_at_start_c": round(max(intercept, 0.0), 2),
        "drift_at_end_c": round(max(intercept + slope * end_day, 0.0), 2),
        "rows_flagged": int(mask.sum()),
    }
    return df


# --------------------------------------------------------------------------------------
# Rule 4: keep extremes, only add diagnostics
# --------------------------------------------------------------------------------------
def add_extreme_diagnostics(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    is_vehicle = df["sensor_id"].isin(VEHICLE_SENSORS)
    df["is_extreme_cold"] = df["clean_c"] < COLD_LIMIT_C
    df["is_extreme_hot"] = (~is_vehicle) & (df["clean_c"] > HOT_LIMIT_C)

    # Rolling median inside gap-free segments only (so rows removed by rules 1-2 do not
    # create fake jumps by gluing separate parts of a series together).
    gap = df.groupby("sensor_id")["timestamp"].diff().ne(STEP)
    seg = gap.groupby(df["sensor_id"]).cumsum()
    med = (
        df.groupby([df["sensor_id"], seg])["clean_c"]
        .transform(lambda s: s.rolling(MEDIAN_WINDOW, center=True, min_periods=1).median())
    )
    df["suspect_isolated_dip"] = (df["clean_c"] - med).abs() >= SPIKE_DELTA

    def _run_hours(g: pd.DataFrame, col: str) -> list[float]:
        """Lengths (hours) of consecutive-reading runs where col is True."""
        t = g.loc[g[col], "timestamp"]
        if t.empty:
            return []
        run_id = (t.diff() != STEP).cumsum()
        return ((t.groupby(run_id).size()) * STEP.total_seconds() / 3600).tolist()

    for col in ("is_extreme_cold", "is_extreme_hot"):
        hours: list[float] = []
        for _, g in df[df[col]].groupby("sensor_id"):
            hours += _run_hours(g.assign(**{col: True}), col)
        report[col] = {
            "rows": int(df[col].sum()),
            "events": len(hours),
            "median_event_hours": round(float(np.median(hours)), 2) if hours else 0.0,
            "max_event_hours": round(float(max(hours)), 2) if hours else 0.0,
            "rows_also_suspect_isolated_dip": int((df[col] & df["suspect_isolated_dip"]).sum()),
        }
    report["suspect_isolated_dip_rows"] = int(df["suspect_isolated_dip"].sum())
    return df


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def prepare(path: str | Path) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    df = load_raw(path)
    df = apply_basic_filters(df, report)
    df = add_u07_drift_columns(df, report)
    df = add_extreme_diagnostics(df, report)
    return df, report


def print_report(df: pd.DataFrame, report: dict) -> None:
    line = "-" * 72
    print(line)
    print("RULE 1 - vehicles (on_board == True only)")
    print(f"  vehicle rows total      : {report['rule1_vehicle_rows_total']:>9,}")
    print(f"  vehicle rows kept       : {report['rule1_vehicle_rows_on_board']:>9,} "
          f"({report['rule1_vehicle_rows_on_board'] / report['rule1_vehicle_rows_total']:.1%})")
    print(f"  rows removed            : {report['rule1_rows_removed']:>9,}")
    print("RULE 2 - empty clean_c")
    print(f"  rows removed            : {report['rule2_rows_removed']:>9,}  {report['rule2_nan_by_sensor']}")
    print(f"ROWS: {report['rows_in']:,} -> {report['rows_out']:,}")
    print(line)
    print("RULE 3 - TS-U07 drift (estimate, original clean_c kept)")
    print(f"  {report['u07']}")
    print(line)
    print("RULE 4 - extremes kept (diagnostics only)")
    print(f"  cold (< {COLD_LIMIT_C:g} C): {report['is_extreme_cold']}")
    print(f"  hot  (> {HOT_LIMIT_C:g} C): {report['is_extreme_hot']}")
    print(f"  suspect_isolated_dip rows: {report['suspect_isolated_dip_rows']}")
    print(line)
    print(f"clean_c range after filtering: {df['clean_c'].min():.1f} .. {df['clean_c'].max():.1f} C")
    print(line)


def _default_input() -> str:
    for name in DEFAULT_INPUTS:
        if Path(name).exists():
            return name
    return DEFAULT_INPUTS[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare cold-chain sensor readings.")
    ap.add_argument("--input", default=_default_input(), help="sensor_readings_clean.csv or .gz")
    ap.add_argument("--output", default="sensor_readings_prepared.csv", help="output CSV path")
    args = ap.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}")

    df, report = prepare(args.input)
    print_report(df, report)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}  ({len(df):,} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
