"""
Cleaning pipeline for the ENG499 cold chain sensor data (IoT temperature loggers).

Steps
  1. share of missing readings and longest gaps per sensor
  2. spike removal (rolling median filter)
  3. linear interpolation of short gaps; long gaps stay empty and are flagged
  4. detection of abnormal sensors (outage, drift, calibration step) with start date
  5. consistency checks on the cleaned table

Input (DATA_DIR)
  iot/sensor_readings.csv.gz
  master/sensors.csv

Output (OUT_DIR)
  sensor_readings_clean.csv.gz   cleaned readings, one row per sensor and 5-minute slot
  sensor_quality_summary.csv     one row per sensor
  cleaning_log.txt               findings and results of the checks
  *.png                          figures

Column "quality" of the clean file
  ok             measured value, unchanged
  spike_removed  measured value was a spike, replaced by an estimate
  interpolated   reading lost in a short gap (30 min or less), linear interpolation
  gap            reading lost in a longer gap, left empty and flagged
  imputed        only with FILL_LONG_GAPS = True

Usage
  1. set DATA_DIR to the student_data folder
  2. python sensor_cleaning.py

Requires pandas, numpy and matplotlib.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings

# Path to the dataset folder (student_data): absolute, or relative to this file.
# The raw data is not stored in the repository, so set it on each machine.
DATA_DIR = "student_data"

# Results are written to this folder (absolute, or relative to this file).
OUT_DIR = "output_clean"

START = pd.Timestamp("2025-04-01 00:00")   # analysis period; the 25-31 March warm-up is excluded
END = pd.Timestamp("2025-09-30 23:55")
STEP = pd.Timedelta(minutes=5)

SPIKE_WINDOW = 7            # readings in the rolling median (35 min)
# A reading farther than this from the rolling median is a spike. On the fixed
# sensors normal readings stay within about 3 C of the median and spikes start at
# about 7 C. The vehicle loggers vary more in normal use (doors, driving).
SPIKE_THRESHOLD_FIXED_C = 5.0
SPIKE_THRESHOLD_VEHICLE_C = 8.0
INTERP_MAX_SAMPLES = 6      # gaps up to 30 min are filled with a straight line

# Optional steps, disabled by default.
FILL_LONG_GAPS = False      # True: fill gaps longer than 30 min with the typical daily value
CORRECT_FAULTS = False      # True: subtract the detected drift / step from the sensor

PROFILE_DAYS = 7            # FILL_LONG_GAPS: median of the same time of day, +-7 days
PROFILE_WEEKS = 4           # FILL_LONG_GAPS, vehicles: same weekday and time of day, +-4 weeks
LONG_GAP_MIN = 15           # for counting "long" gaps in the summary
OUTAGE_MIN = 24 * 60        # a gap of 24 h or more is reported as an outage
FLAG_SIZE_C = 0.6           # drifts / steps smaller than this are ignored
MIN_EXPLAINED = 0.5         # the change must explain at least half of the variation
MIN_PEERS_TO_CORRECT = 3    # only correct a sensor if its group has 3+ sensors

QUALITY_VALUES = ["ok", "spike_removed", "interpolated", "gap", "imputed"]

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                 # __file__ is not defined in an interactive console
    SCRIPT_DIR = os.getcwd()


def resolve(path):
    """Resolve a relative path against the folder of this script."""
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


DATA_DIR = resolve(DATA_DIR)
OUT_DIR = resolve(OUT_DIR)
GRID = pd.date_range(START, END, freq="5min")


# ----------------------------------------------------------------- loading
def load_readings():
    df = pd.read_csv(os.path.join(DATA_DIR, "iot", "sensor_readings.csv.gz"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "timestamp"])
    df = df.drop_duplicates(["sensor_id", "timestamp"])
    return df


def load_sensor_table():
    sensors = pd.read_csv(os.path.join(DATA_DIR, "master", "sensors.csv"))
    sites_path = os.path.join(DATA_DIR, "master", "sites.csv")
    if os.path.exists(sites_path):     # optional: cabinet class, used to group comparable sensors
        sites = pd.read_csv(sites_path)[["site_id", "cabinet_class"]]
        sensors = sensors.merge(sites, left_on="asset_id", right_on="site_id", how="left")
        sensors["cabinet_class"] = sensors["cabinet_class"].fillna("")
    else:
        sensors["cabinet_class"] = ""
    # Sensors in the same group are compared with each other.
    # The two cold room sensors (front / back) form one group.
    sensors["group"] = np.where(sensors["placement"].str.startswith("cold_room"),
                                "cold_room", sensors["placement"] + "_" + sensors["cabinet_class"])
    return sensors


def load_onboard_windows():
    """Time windows in which crates are on board each vehicle (from the event log)."""
    events = pd.read_csv(os.path.join(DATA_DIR, "traceability", "events.csv.gz"))
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    units = pd.read_csv(os.path.join(DATA_DIR, "traceability", "shipment_units.csv"))
    units = units[["su_id", "vehicle_id"]]
    events = events[events["event"].isin(["loaded_on_vehicle", "unloading_started"])]
    events = events.merge(units, on="su_id", how="left").dropna(subset=["vehicle_id"])
    events["day"] = events["timestamp"].dt.date
    win = events.groupby(["vehicle_id", "day"])["timestamp"].agg(["min", "max"]).reset_index()
    win["max"] = win["max"] + pd.Timedelta(minutes=30)     # margin for the end of unloading
    return win.rename(columns={"min": "start", "max": "end"})


def onboard_mask(index, vehicle_id, windows):
    keep = pd.Series(False, index=index)
    for _, row in windows[windows["vehicle_id"] == vehicle_id].iterrows():
        keep.loc[row["start"]:row["end"]] = True
    return keep


# ---------------------------------------------- 1. missing readings, gaps
def gap_report(readings):
    expected = len(GRID)
    rows = []
    for sid, g in readings.groupby("sensor_id"):
        ts = g.loc[(g["timestamp"] >= START) & (g["timestamp"] <= END), "timestamp"]
        ts = ts.reset_index(drop=True)
        # sentinel timestamps before the first and after the last slot, so gaps at
        # the start or end of the period are detected too
        t = pd.concat([pd.Series([START - STEP]), ts, pd.Series([END + STEP])],
                      ignore_index=True)
        missing_min = t.diff().dt.total_seconds() / 60.0 - 5.0
        i = missing_min.idxmax()
        rows.append({
            "sensor_id": sid,
            "expected_readings": expected,
            "actual_readings": len(ts),
            "missing_pct": round(100.0 * (1 - len(ts) / expected), 2),
            "longest_gap_min": int(missing_min.max()),
            "longest_gap_start": t[i - 1] + STEP,
            "longest_gap_end": t[i] - STEP,
            "n_gaps_ge_15min": int((missing_min >= LONG_GAP_MIN).sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 2. spikes
def remove_spikes(series, threshold):
    """Rolling median filter. Returns the series with spikes set to NaN and a boolean spike flag."""
    med = series.rolling(SPIKE_WINDOW, center=True, min_periods=3).median()
    is_spike = (series - med).abs() > threshold
    return series.mask(is_spike), is_spike


# ------------------------------------------------------- 3. filling gaps
def fill_short_gaps(s, max_samples=INTERP_MAX_SAMPLES):
    """
    Linear interpolation of gaps with at most max_samples consecutive missing values.
    Longer gaps are left unchanged (Series.interpolate(limit=...) would fill their
    first part). Returns the filled series and a mask of the filled slots.
    """
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    line = s.interpolate(method="time", limit_area="inside")
    short = isna & (run_len <= max_samples) & line.notna()
    return s.where(~short, line), short


def profile_fill(s, vehicle):
    """
    Fill remaining NaN with the median of the same sensor at the same time of day on
    neighbouring days (same weekday for vehicles, whose routes depend on the weekday).
    """
    first_day = s.index[0].normalize()
    days = (s.index.normalize() - first_day).days.to_numpy()
    slots = (s.index.hour * 12 + s.index.minute // 5).to_numpy()
    arr = np.full((days.max() + 1, 288), np.nan)
    arr[days, slots] = s.to_numpy()
    mat = pd.DataFrame(arr)
    if vehicle:
        weekday = (first_day + pd.to_timedelta(mat.index, unit="D")).dayofweek
        prof = mat.groupby(weekday).transform(
            lambda x: x.rolling(2 * PROFILE_WEEKS + 1, center=True, min_periods=1).median())
    else:
        prof = mat.rolling(2 * PROFILE_DAYS + 1, center=True, min_periods=1).median()
    typical = pd.Series(prof.to_numpy()[days, slots], index=s.index)
    out = s.fillna(typical)
    # fallback if the whole neighbourhood is empty
    return out.interpolate(method="time", limit_direction="both")


# ------------------------------------------ 4. abnormal sensors (drift...)
def find_change(res, min_seg=14):
    """
    Detect a step or a drift in a daily residual series (sensor minus group median).
    A step model and a flat-then-ramp model are fitted for every candidate start day;
    the model with the smallest squared error is returned.
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
                    "size_c": post.mean() - pre.mean(), "slope_per_day": 0.0}
        ramp = np.where(x > x[k], x - x[k], 0.0)
        A = np.column_stack([np.ones(len(x)), ramp])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        sse_ramp = ((y - A @ coef) ** 2).sum()
        if sse_ramp < best["sse"]:
            best = {"type": "drift", "sse": sse_ramp, "k": k,
                    "size_c": coef[1] * (x[-1] - x[k]), "slope_per_day": coef[1]}
    best["explained"] = 1 - best["sse"] / sse_flat
    best["start"] = r.index[best["k"]]
    return best


def daily_residuals(daily, sensors):
    """Daily median of every sensor minus the median of its group."""
    group_of = dict(zip(sensors["sensor_id"], sensors["group"]))
    res = pd.DataFrame(index=daily.index)
    n_peers = {}
    for sid in daily.columns:
        peers = [s for s in daily.columns if group_of.get(s) == group_of.get(sid)]
        n_peers[sid] = len(peers)
        res[sid] = daily[sid] - daily[peers].median(axis=1)
    return res, n_peers


def significant(ch):
    return (ch is not None and abs(ch["size_c"]) >= FLAG_SIZE_C
            and ch["explained"] >= MIN_EXPLAINED)


def correction_for(index, ch):
    """Correction in degrees C for every time stamp (0 before the change starts)."""
    days = (index - ch["start"]).total_seconds().to_numpy() / 86400.0
    if ch["type"] == "drift":
        return np.clip(days, 0, None) * ch["slope_per_day"]
    return np.where(days >= 0, ch["size_c"], 0.0)


# ---------------------------------------------------------- one sensor
def prepare_sensor(g, vehicle):
    """Spike removal and short-gap interpolation for one sensor (temperature, humidity, battery)."""
    raw = g.set_index("timestamp").reindex(GRID)
    temp_raw = raw["temperature_c"]
    threshold = SPIKE_THRESHOLD_VEHICLE_C if vehicle else SPIKE_THRESHOLD_FIXED_C
    no_spike, is_spike = remove_spikes(temp_raw, threshold)
    temp, interp_mask = fill_short_gaps(no_spike)

    # humidity and battery use the same gap rule as the temperature
    hum, _ = fill_short_gaps(raw["humidity_pct"])
    bat, _ = fill_short_gaps(raw["battery_pct"])
    if FILL_LONG_GAPS:
        hum = profile_fill(hum, vehicle)
        bat = bat.interpolate(method="time", limit_direction="both")
    return {"raw": temp_raw, "temp": temp, "spike": is_spike, "interp": interp_mask,
            "hum": hum.round(1), "bat": bat.round(0)}


# -------------------------------------------------------------- figures
def make_figures(summary, before, after, clean_all):
    plt.figure(figsize=(10, 4))
    plt.bar(summary["sensor_id"], summary["missing_pct"])
    plt.xticks(rotation=90)
    plt.ylabel("missing readings (%)")
    plt.title("Missing readings per sensor, 1 Apr - 30 Sep 2025")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "missing_per_sensor.png"), dpi=150)
    plt.close()

    for _, row in summary[summary["change_type"] != ""].iterrows():
        sid = row["sensor_id"]
        plt.figure(figsize=(9, 3.5))
        before[sid].plot(label="before correction" if row["corrected"] else "difference to peers")
        if row["corrected"]:
            after[sid].plot(label="after correction")
        plt.axvline(row["change_start"], color="red", linestyle="--")
        plt.ylabel("sensor minus peer median (C)")
        plt.title("%s - %s from %s" % (sid, row["change_type"], str(row["change_start"])[:10]))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "abnormal_%s.png" % sid), dpi=150)
        plt.close()

    # figure of the longest outage (raw and cleaned data)
    worst = summary.sort_values("longest_gap_min").iloc[-1]
    if worst["longest_gap_min"] >= OUTAGE_MIN:
        sid = worst["sensor_id"]
        g = clean_all[clean_all["sensor_id"] == sid].set_index("timestamp")
        a = worst["longest_gap_start"] - pd.Timedelta(days=2)
        b = worst["longest_gap_end"] + pd.Timedelta(days=2)
        g = g.loc[a:b]
        plt.figure(figsize=(9, 3.2))
        plt.plot(g.index, g["temperature_c"], label="clean", color="tab:green")
        plt.plot(g.index, g["raw_temperature_c"], label="raw", color="tab:blue", lw=0.8)
        plt.legend()
        plt.ylabel("C")
        plt.title("%s: longest outage" % sid)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "outage_%s.png" % sid), dpi=150)
        plt.close()


# ------------------------------------------------------------ checks
def validate(clean, log):
    """Consistency checks on the clean table. Returns False if any check fails."""
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        log("[%s] %s %s" % ("PASS" if passed else "FAIL", name, detail))

    n_sensors = clean["sensor_id"].nunique()
    gap_rows = clean["quality"] == "gap"
    per_sensor = clean.groupby("sensor_id").size()
    check("every sensor has all %d time slots" % len(GRID), bool((per_sensor == len(GRID)).all()),
          "(%d sensors)" % n_sensors)
    check("no duplicate (sensor, time)", not clean.duplicated(["sensor_id", "timestamp"]).any())
    for col in ["timestamp", "sensor_id", "asset_id", "placement", "quality"]:
        check("no empty values in %s" % col, int(clean[col].isna().sum()) == 0)
    # temperature / humidity / battery may be empty only in rows marked "gap"
    for col in ["temperature_c", "humidity_pct", "battery_pct"]:
        bad = int((clean[col].isna() & ~gap_rows).sum())
        check("%s is empty only in 'gap' rows" % col, bad == 0, "(%d rows outside gaps)" % bad)
    check("a 'gap' row has no temperature", bool(clean.loc[gap_rows, "temperature_c"].isna().all()))
    check("quality has only allowed values", set(clean["quality"].unique()) <= set(QUALITY_VALUES))
    fixed = clean[~clean["placement"].eq("vehicle_reefer")]["temperature_c"].dropna()
    check("fixed sensors between -5 and 25 C", bool(fixed.between(-5, 25).all()),
          "(min %.1f, max %.1f)" % (fixed.min(), fixed.max()))
    check("humidity between 0 and 100", bool(clean["humidity_pct"].dropna().between(0, 100).all()))
    check("battery between 0 and 100", bool(clean["battery_pct"].dropna().between(0, 100).all()))
    return ok


# ------------------------------------------------------------------- main
def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit("Could not find the data folder: %s\n"
                         "Set DATA_DIR at the top of this file." % DATA_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    readings = load_readings()
    sensors = load_sensor_table().set_index("sensor_id")

    # readings off the 5-minute grid would be dropped by the reindexing
    inside = readings[(readings["timestamp"] >= START) & (readings["timestamp"] <= END)]
    off_grid = int((inside["timestamp"].dt.minute % 5 != 0).sum())
    log("readings in the period: %d (%d not on the 5-minute grid)" % (len(inside), off_grid))

    log("\n1) missing readings and gaps ...")
    gaps = gap_report(readings)

    log("2) spikes and short gaps ...")
    try:
        windows = load_onboard_windows()
    except Exception as e:
        log("   (no on-board windows for the vehicles: %s)" % e)
        windows = None

    prep, onboard, daily = {}, {}, {}
    for sid, g in readings.groupby("sensor_id"):
        placement = sensors.loc[sid, "placement"]
        vehicle = placement == "vehicle_reefer"
        prep[sid] = prepare_sensor(g, vehicle)
        analysis = prep[sid]["temp"]
        if vehicle:
            if windows is not None:
                onboard[sid] = onboard_mask(GRID, sensors.loc[sid, "asset_id"], windows)
                analysis = analysis.where(onboard[sid])
            else:                       # parked vehicles log the outdoor temperature
                analysis = pd.Series(np.nan, index=GRID)
        d = analysis.resample("D").agg(["median", "count"])
        daily[sid] = d["median"].where(d["count"] >= 12)
    daily = pd.DataFrame(daily)

    log("3) sensors that behave abnormally ...")
    res_before, n_peers = daily_residuals(daily, sensors.reset_index())
    changes = {sid: find_change(res_before[sid]) for sid in daily.columns}
    corrections = {}
    for sid, ch in changes.items():
        if significant(ch):
            if not CORRECT_FAULTS:
                log("   %s: %s from %s, size %.2f C (explains %.0f %% of the variation) "
                    "-> reported, not corrected" % (sid, ch["type"], str(ch["start"])[:10],
                                                    ch["size_c"], 100 * ch["explained"]))
            elif n_peers[sid] >= MIN_PEERS_TO_CORRECT:
                corrections[sid] = correction_for(GRID, ch)
                log("   %s: %s from %s, size %.2f C -> corrected" % (
                    sid, ch["type"], str(ch["start"])[:10], ch["size_c"]))
            else:
                log("   %s: %s from %s, size %.2f C -> NOT corrected (only %d sensors in its "
                    "group, cannot tell which one is wrong)" % (
                        sid, ch["type"], str(ch["start"])[:10], ch["size_c"], n_peers[sid]))

    log("4) building the clean table (fill long gaps: %s, correct faults: %s) ..." % (
        FILL_LONG_GAPS, CORRECT_FAULTS))
    parts = []
    daily_after = daily.copy()
    for sid, p in prep.items():
        placement = sensors.loc[sid, "placement"]
        vehicle = placement == "vehicle_reefer"
        corr = pd.Series(corrections.get(sid, np.zeros(len(GRID))), index=GRID)
        temp = p["temp"] - corr                      # NaN stays NaN
        still_empty = temp.isna()
        quality = pd.Series("ok", index=GRID)
        quality[p["interp"]] = "interpolated"
        if FILL_LONG_GAPS:
            temp = profile_fill(temp, vehicle)       # optional: long gaps
            quality[still_empty] = "imputed"
            quality[p["spike"]] = "spike_removed"
        else:                                        # long gaps stay empty and are flagged
            quality[p["spike"]] = "spike_removed"
            quality[still_empty] = "gap"

        if sid in corrections:                       # recompute daily medians to verify the correction
            a = temp.where(onboard[sid]) if sid in onboard else temp
            d = a.resample("D").agg(["median", "count"])
            daily_after[sid] = d["median"].where(d["count"] >= 12)

        out = pd.DataFrame({
            "timestamp": GRID,
            "sensor_id": sid,
            "asset_id": sensors.loc[sid, "asset_id"],
            "placement": placement,
            "temperature_c": temp.round(2).to_numpy(),
            "raw_temperature_c": p["raw"].to_numpy(),
            "humidity_pct": p["hum"].to_numpy(),
            "battery_pct": p["bat"].to_numpy(),
            "quality": quality.to_numpy(),
            "on_board": onboard[sid].to_numpy() if sid in onboard else np.nan,
            "correction_c": corr.round(2).to_numpy(),
        })
        parts.append(out)
    clean = pd.concat(parts, ignore_index=True)
    clean = clean.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    res_after, _ = daily_residuals(daily_after, sensors.reset_index())

    # summary table, one row per sensor
    rows = []
    for _, gp in gaps.iterrows():
        sid = gp["sensor_id"]
        ch = changes[sid]
        flagged = significant(ch)
        after = find_change(res_after[sid]) if sid in corrections else None
        q = clean.loc[clean["sensor_id"] == sid, "quality"].value_counts()
        rows.append({
            **gp.to_dict(),
            "placement": sensors.loc[sid, "placement"],
            "asset_id": sensors.loc[sid, "asset_id"],
            "n_spikes_removed": int(q.get("spike_removed", 0)),
            "n_interpolated": int(q.get("interpolated", 0)),
            "n_gap_rows": int(q.get("gap", 0)),
            "n_imputed": int(q.get("imputed", 0)),
            "outage": "yes" if gp["longest_gap_min"] >= OUTAGE_MIN else "",
            "change_type": ch["type"] if flagged else "",
            "change_start": ch["start"] if flagged else pd.NaT,
            "change_size_c": round(ch["size_c"], 2) if flagged else np.nan,
            "explained": round(ch["explained"], 2) if flagged else np.nan,
            "corrected": sid in corrections,
            "size_after_correction_c": round(after["size_c"], 2) if after is not None else np.nan,
        })
    summary = pd.DataFrame(rows)

    log("\n5) checks ...")
    passed = validate(clean, log)

    clean_out = clean.copy()
    clean_out["timestamp"] = clean_out["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    clean_out.to_csv(os.path.join(OUT_DIR, "sensor_readings_clean.csv.gz"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "sensor_quality_summary.csv"), index=False)
    make_figures(summary, res_before, res_after, clean)

    log("\nquality of the %d clean rows:" % len(clean))
    log(clean["quality"].value_counts().to_string())
    log("\nper sensor:")
    cols = ["sensor_id", "missing_pct", "longest_gap_min", "n_spikes_removed", "n_gap_rows",
            "outage", "change_type", "change_start", "change_size_c"]
    log(summary[cols].to_string(index=False))
    with open(os.path.join(OUT_DIR, "cleaning_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("\nResults saved in %s" % OUT_DIR)
    if not passed:
        raise SystemExit("Some checks FAILED, do not use the clean file.")


if __name__ == "__main__":
    main()
