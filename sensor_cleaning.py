"""
ENG499 - CSE-1 : sensor data quality analysis + cleaning

Usage (from the folder that contains student_data/):
    python sensor_cleaning.py

What it does
    1. Missing share + longest gaps per sensor
    2. Spike removal (rolling median) and short-gap interpolation
    3. Abnormal-sensor detection (outage, drift, calibration step)
    4. Saves tables, figures and a cleaned readings file to output_quality/

Needs: pandas, numpy, matplotlib
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- settings
# Paths are built from the location of THIS file, so it works from Spyder,
# VS Code or the terminal no matter what the working directory is.
# Put this file in the same folder as student_data/ and quickstart.py.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:               # e.g. pasted into a console
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "student_data")
OUT_DIR = os.path.join(BASE_DIR, "output_quality")

START = pd.Timestamp("2025-04-01 00:00")   # warm-up (25-31 March) excluded
END = pd.Timestamp("2025-09-30 23:55")
STEP = pd.Timedelta(minutes=5)

SPIKE_WINDOW = 7          # samples in the rolling median (35 min)
SPIKE_THRESHOLD_C = 10.0  # |value - rolling median| above this = spike (real spikes: 10-25 C)
MAX_INTERP_SAMPLES = 6    # interpolate gaps up to 30 min, flag longer ones
LONG_GAP_MIN = 15         # gaps >= this are counted as "long"
OUTAGE_MIN = 24 * 60      # a gap >= 24 h is treated as logger offline
FLAG_SIZE_C = 0.6         # drift / step smaller than this is not flagged
MIN_EXPLAINED = 0.5       # change model must explain >= 50 % of the variance


# ------------------------------------------------------------------ loading
def load_readings():
    path = os.path.join(DATA_DIR, "iot", "sensor_readings.csv.gz")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "timestamp"])
    df = df.drop_duplicates(["sensor_id", "timestamp"])
    return df


def load_sensor_table():
    sensors = pd.read_csv(os.path.join(DATA_DIR, "master", "sensors.csv"))
    sites_path = os.path.join(DATA_DIR, "master", "sites.csv")
    if os.path.exists(sites_path):           # optional: only used to group cabinets
        sites = pd.read_csv(sites_path)[["site_id", "cabinet_class"]]
        sensors = sensors.merge(sites, left_on="asset_id", right_on="site_id", how="left")
        sensors["cabinet_class"] = sensors["cabinet_class"].fillna("")
    else:
        sensors["cabinet_class"] = ""
    # sensors that must be compared with each other
    sensors["group"] = sensors["placement"] + "_" + sensors["cabinet_class"]
    return sensors


def load_onboard_windows():
    """Time windows in which crates are really inside each vehicle."""
    events = pd.read_csv(os.path.join(DATA_DIR, "traceability", "events.csv.gz"))
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    units = pd.read_csv(os.path.join(DATA_DIR, "traceability", "shipment_units.csv"))
    units = units[["su_id", "vehicle_id"]]
    events = events[events["event"].isin(["loaded_on_vehicle", "unloading_started"])]
    events = events.merge(units, on="su_id", how="left").dropna(subset=["vehicle_id"])
    events["day"] = events["timestamp"].dt.date
    win = events.groupby(["vehicle_id", "day"])["timestamp"].agg(["min", "max"])
    win = win.reset_index()
    win["max"] = win["max"] + pd.Timedelta(minutes=30)   # time to finish unloading
    win = win.rename(columns={"min": "start", "max": "end"})
    return win


# ---------------------------------------------------- 1. missing and gaps
def gap_report(readings):
    expected = len(pd.date_range(START, END, freq="5min"))
    rows = []
    for sid, g in readings.groupby("sensor_id"):
        ts = g.loc[(g["timestamp"] >= START) & (g["timestamp"] <= END), "timestamp"]
        ts = ts.reset_index(drop=True)
        # sentinels so gaps at the very start / end are counted too
        t = pd.concat([pd.Series([START - STEP]), ts, pd.Series([END + STEP])],
                      ignore_index=True)
        missing_min = t.diff().dt.total_seconds() / 60.0 - 5.0
        i = missing_min.idxmax()
        battery = g["battery_pct"].iloc[-1] if "battery_pct" in g.columns else np.nan
        rows.append({
            "sensor_id": sid,
            "expected_readings": expected,
            "actual_readings": len(ts),
            "missing_pct": round(100.0 * (1 - len(ts) / expected), 2),
            "longest_gap_min": int(missing_min.max()),
            "longest_gap_start": t[i - 1] + STEP,
            "longest_gap_end": t[i] - STEP,
            "n_gaps_ge_15min": int((missing_min >= LONG_GAP_MIN).sum()),
            "last_reading": ts.iloc[-1] if len(ts) > 0 else pd.NaT,
            "last_battery_pct": battery,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------- 2. cleaning
def remove_spikes(series):
    """Rolling-median filter. Returns (series with spikes set to NaN, spike flags)."""
    med = series.rolling(SPIKE_WINDOW, center=True, min_periods=3).median()
    is_spike = (series - med).abs() > SPIKE_THRESHOLD_C
    return series.mask(is_spike), is_spike


def clean_sensor(g):
    """Clean one sensor. g = rows of one sensor_id. Returns a DataFrame on a 5-min grid."""
    grid = pd.date_range(START, END, freq="5min")
    raw = g.set_index("timestamp")["temperature_c"].reindex(grid)
    no_spike, is_spike = remove_spikes(raw)
    filled = no_spike.interpolate(method="time", limit=MAX_INTERP_SAMPLES,
                                  limit_area="inside")
    out = pd.DataFrame({"raw_c": raw, "clean_c": filled})
    out["is_spike"] = is_spike
    out["is_missing"] = raw.isna()
    out["is_interpolated"] = filled.notna() & no_spike.isna()
    out["is_long_gap"] = filled.isna()          # still empty: do NOT assume "no problem"
    return out


def onboard_mask(index, vehicle_id, windows):
    keep = pd.Series(False, index=index)
    w = windows[windows["vehicle_id"] == vehicle_id]
    for _, row in w.iterrows():
        keep.loc[row["start"]:row["end"]] = True
    return keep


# --------------------------------------------- 3. abnormal sensor detection
def find_change(res, min_seg=14):
    """
    res = daily residual of a sensor (sensor - median of its peer group).
    Tries a step model and a drift model (flat, then linear ramp) at every
    possible start day and keeps the one with the smallest error.
    """
    r = res.dropna()
    if len(r) < 3 * min_seg:
        return None
    x = (r.index - r.index[0]).days.values.astype(float)
    y = r.values.astype(float)
    sse_flat = ((y - y.mean()) ** 2).sum()
    if sse_flat == 0:
        return None
    best = None
    for k in range(min_seg, len(y) - min_seg):
        pre, post = y[:k], y[k:]
        sse_step = ((pre - pre.mean()) ** 2).sum() + ((post - post.mean()) ** 2).sum()
        if best is None or sse_step < best["sse"]:
            best = {"type": "step", "sse": sse_step, "k": k,
                    "size_c": post.mean() - pre.mean(), "per_month_c": np.nan}
        ramp = np.where(x > x[k], x - x[k], 0.0)
        A = np.column_stack([np.ones(len(x)), ramp])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        sse_ramp = ((y - A @ coef) ** 2).sum()
        if sse_ramp < best["sse"]:
            best = {"type": "drift", "sse": sse_ramp, "k": k,
                    "size_c": coef[1] * (x[-1] - x[k]), "per_month_c": coef[1] * 30}
    best["explained"] = 1 - best["sse"] / sse_flat
    best["start"] = r.index[best["k"]]
    return best


def detect_abnormal(daily, sensors, gaps):
    """daily: DataFrame (index = date, columns = sensor_id) of daily medians."""
    group_of = dict(zip(sensors["sensor_id"], sensors["group"]))
    residuals = pd.DataFrame(index=daily.index)
    for sid in daily.columns:
        peers = [s for s in daily.columns if group_of.get(s) == group_of.get(sid)]
        residuals[sid] = daily[sid] - daily[peers].median(axis=1)

    gap_by_sensor = gaps.set_index("sensor_id")
    rows = []
    for sid in daily.columns:
        issue, start, size, how = "", pd.NaT, np.nan, ""
        gp = gap_by_sensor.loc[sid]
        if gp["longest_gap_min"] >= OUTAGE_MIN:
            issue = "offline / long outage"
            start = gp["longest_gap_start"]
            size = gp["longest_gap_min"] / 60.0
            how = "gap of %.1f h in the timestamps (battery at end: %s)" % (
                size, gp["last_battery_pct"])
        elif daily[sid].notna().sum() < 30:
            issue = "not assessed (not enough usable data, e.g. no event log for vehicles)"
        else:
            ch = find_change(residuals[sid])
            if (ch is not None and abs(ch["size_c"]) >= FLAG_SIZE_C
                    and ch["explained"] >= MIN_EXPLAINED):
                issue = "calibration step" if ch["type"] == "step" else "slow drift"
                start = ch["start"]
                size = ch["size_c"]
                how = "residual vs peers changes by %.2f C (explained %.0f %% of variance)" % (
                    size, 100 * ch["explained"])
        rows.append({"sensor_id": sid, "issue": issue, "issue_start": start,
                     "size": size, "how_detected": how})
    return pd.DataFrame(rows), residuals


# --------------------------------------------------------------- figures
def make_figures(summary, residuals):
    plt.figure(figsize=(10, 4))
    plt.bar(summary["sensor_id"], summary["missing_pct"])
    plt.xticks(rotation=90)
    plt.ylabel("missing readings (%)")
    plt.title("Missing readings per sensor, 1 Apr - 30 Sep 2025")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "missing_per_sensor.png"), dpi=150)
    plt.close()

    flagged = summary[(summary["issue"] != "") & (~summary["issue"].str.startswith("offline")) & (~summary["issue"].str.startswith("not assessed"))]
    for _, row in flagged.iterrows():
        plt.figure(figsize=(9, 3.5))
        residuals[row["sensor_id"]].plot()
        plt.axvline(row["issue_start"], color="red", linestyle="--")
        plt.ylabel("sensor minus peer median (C)")
        plt.title("%s - %s (start %s)" % (row["sensor_id"], row["issue"],
                                         str(row["issue_start"])[:10]))
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "abnormal_%s.png" % row["sensor_id"]), dpi=150)
        plt.close()


# ------------------------------------------------------------------ main
def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit("student_data folder not found at: %s\n"
                         "Move this file next to the student_data folder." % DATA_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    readings = load_readings()
    sensors = load_sensor_table()

    print("1) gaps and missing share ...")
    gaps = gap_report(readings)

    print("2) cleaning ...")
    try:
        windows = load_onboard_windows()
    except Exception as e:
        print("   (could not build on-board windows for vehicles: %s)" % e)
        windows = None

    cleaned_parts = []
    daily = {}
    n_spikes = {}
    for sid, g in readings.groupby("sensor_id"):
        out = clean_sensor(g)
        n_spikes[sid] = int(out["is_spike"].sum())
        analysis = out["clean_c"]
        placement = sensors.loc[sensors["sensor_id"] == sid, "placement"].iloc[0]
        asset = sensors.loc[sensors["sensor_id"] == sid, "asset_id"].iloc[0]
        if placement == "vehicle_reefer":
            if windows is not None:
                keep = onboard_mask(out.index, asset, windows)
                out["on_board"] = keep
                analysis = analysis.where(keep)
            else:
                # parked-van readings (outdoor temperature) would give false alarms
                analysis = pd.Series(np.nan, index=analysis.index)
        # daily median, only if the day has at least 1 hour of data
        d = analysis.resample("D").agg(["median", "count"])
        daily[sid] = d["median"].where(d["count"] >= 12)
        out.insert(0, "sensor_id", sid)
        cleaned_parts.append(out.reset_index().rename(columns={"index": "timestamp"}))
    daily = pd.DataFrame(daily)

    print("3) abnormal sensors ...")
    abnormal, residuals = detect_abnormal(daily, sensors, gaps)

    summary = gaps.merge(sensors[["sensor_id", "placement", "asset_id"]], on="sensor_id")
    summary = summary.merge(abnormal, on="sensor_id")
    summary["n_spikes_removed"] = summary["sensor_id"].map(n_spikes)
    summary.to_csv(os.path.join(OUT_DIR, "sensor_quality_summary.csv"), index=False)

    pd.concat(cleaned_parts).to_csv(
        os.path.join(OUT_DIR, "sensor_readings_clean.csv.gz"), index=False)
    make_figures(summary, residuals)

    print()
    cols = ["sensor_id", "missing_pct", "longest_gap_min", "n_spikes_removed", "issue",
            "issue_start"]
    print(summary[cols].to_string(index=False))
    print("\nFiles saved in", OUT_DIR)


if __name__ == "__main__":
    main()
