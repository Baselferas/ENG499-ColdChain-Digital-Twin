"""
ENG499 Cold-Chain Digital Twin - Screen 1 (Week 2, CSE-3) - simple first version
================================================================================
For a selected shipment unit (SU) the screen shows:
  1. Where it was and when   (event log + timeline, monitored vs. blind spot)
  2. Temperature history     (cold room -> vehicle -> store cabinet)
  3. RSL (remaining shelf life) and simple alerts

Run:  python -m streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Cold-Chain Digital Twin", page_icon="🥛", layout="wide")

# Hide the "Deploy" button
st.markdown(
    "<style>[data-testid='stAppDeployButton'], .stAppDeployButton {display: none;}</style>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("COLDCHAIN_DATA", Path(__file__).resolve().parents[1] / "student_data"))

WARMUP_END = pd.Timestamp("2025-04-01")   # 25-31 Mar 2025 is the warm-up week -> excluded
FLOOR_WARN_H = 0.5                        # store-floor wait threshold (hours)
MIN_RSL_DELIVERY = 2.0                    # stores reject crates with RSL < 2 days
SPIKE_DELTA = 6.0                         # |raw - rolling median| >= 6 C -> sensor spike
MEDIAN_WINDOW = 5

STAGE_COLORS = {
    "1 · Depot cold room": "#1f77b4",
    "3 · Vehicle transit": "#2ca02c",
    "4 · Unloading at store": "#ff7f0e",
    "6 · Display cabinet": "#9467bd",
}
COVERAGE_COLORS = {"Monitored": "#2e7d32", "Blind spot": "#c62828"}
CHECKPOINT_ORDER = ["arrival"] + [f"display_day_{i}" for i in range(1, 6)]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _naive(series: pd.Series) -> pd.Series:
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_localize(None)
    return series


def fmt_dur(hours) -> str:
    if hours is None or pd.isna(hours):
        return "n/a"
    minutes = int(round(float(hours) * 60))
    d, rem = divmod(minutes, 1440)
    h, m = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h}h {m:02d}m"


# --------------------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading traceability and ML tables ...")
def load_tables():
    tr = DATA_DIR / "traceability"
    lots = pd.read_csv(tr / "lots.csv")

    su = pd.read_csv(tr / "shipment_units.csv")
    for c in ("delivery_date", "placed_in_display", "last_unit_gone"):
        su[c] = pd.to_datetime(su[c], errors="coerce")

    ev = pd.read_csv(tr / "events.csv.gz")
    ev["timestamp"] = _naive(pd.to_datetime(ev["timestamp"], errors="coerce"))
    ev = ev.sort_values("timestamp").reset_index(drop=True)

    ml = pd.read_csv(DATA_DIR / "ml" / "rsl_training_table.csv")
    ml["checkpoint_time"] = _naive(pd.to_datetime(ml["checkpoint_time"], errors="coerce"))
    ml["target_over_limit"] = ml["target_over_limit"].map(
        lambda v: str(v).strip().lower() in ("1", "1.0", "true")
    )

    sites = pd.read_csv(DATA_DIR / "master" / "sites.csv")
    return lots, su, ev, ml, sites


@st.cache_resource(show_spinner="Loading sensor readings (first run only) ...")
def load_sensors():
    df = pd.read_csv(
        DATA_DIR / "iot" / "sensor_readings.csv.gz",
        usecols=["timestamp", "sensor_id", "asset_id", "temperature_c"],
        dtype={"sensor_id": "category", "asset_id": "category"},
    )
    df["timestamp"] = _naive(pd.to_datetime(df["timestamp"], errors="coerce"))
    df = df.sort_values("timestamp")
    return {str(a): g.reset_index(drop=True) for a, g in df.groupby("asset_id", observed=True)}


def slice_sensor(sensors: dict, asset: str, t0, t1) -> pd.DataFrame:
    """Readings of one asset in [t0, t1)."""
    g = sensors.get(str(asset))
    if g is None:
        return pd.DataFrame()
    lo = g["timestamp"].searchsorted(pd.Timestamp(t0), side="left")
    hi = g["timestamp"].searchsorted(pd.Timestamp(t1), side="left")
    return g.iloc[lo:hi].copy()


@st.cache_data(show_spinner="Building SU list ...")
def build_index(_su, _ev, _ml, _sites):
    """One row per SU with alert flags (used only for the sidebar list)."""
    ex = _ev[_ev["su_id"].notna() & _ev["event"].isin(["received_at_store", "placed_in_display"])]
    ts = ex.groupby(["su_id", "event"])["timestamp"].min().unstack()
    floor_h = ((ts["placed_in_display"] - ts["received_at_store"]).dt.total_seconds() / 3600).rename("floor_h")

    arrival = _ml[_ml["checkpoint"] == "arrival"].set_index("su_id")[["target_rsl_days"]].add_prefix("arr_")
    arrival = arrival[~arrival.index.duplicated(keep="first")]
    agg = _ml.groupby("su_id").agg(max_dh8=("meas_dh_above_8", "max"), any_over=("target_over_limit", "max"))

    sites_small = _sites[["site_id", "cabinet_class"]].set_index("site_id")
    idx = _su.set_index("su_id").join(sites_small, on="store").join(floor_h).join(arrival).join(agg)

    idx["floor_alert"] = (idx["floor_h"] > FLOOR_WARN_H).fillna(False)
    idx["heat_alert"] = idx["max_dh8"].fillna(0) > 0
    idx["red_alert"] = (idx["arr_target_rsl_days"] < MIN_RSL_DELIVERY).fillna(False) | idx["any_over"].fillna(False).astype(bool)
    idx["any_alert"] = idx["floor_alert"] | idx["heat_alert"] | idx["red_alert"]
    idx["normal"] = idx["arr_target_rsl_days"].notna() & ~idx["any_alert"]
    idx["flags"] = (
        np.where(idx["floor_alert"], "🕒", "")
        + np.where(idx["heat_alert"], "🔥", "")
        + np.where(idx["red_alert"], "🔴", "")
    )
    return idx.reset_index()


# --------------------------------------------------------------------------------------
# SU history, stages, temperatures
# --------------------------------------------------------------------------------------
def get_history(su_row, ev: pd.DataFrame) -> pd.DataFrame:
    """Lot-level 'produced_into_cold_room' (su_id is NaN) + all events of this su_id."""
    prod = ev[(ev["lot_id"] == su_row["lot_id"]) & ev["su_id"].isna() & (ev["event"] == "produced_into_cold_room")]
    own = ev[ev["su_id"] == su_row["su_id"]]
    return pd.concat([prod, own]).sort_values("timestamp").reset_index(drop=True)


def build_stages(T: dict, su_row, end_time) -> pd.DataFrame:
    store, veh = su_row["store"], su_row["vehicle_id"]
    defs = [
        ("1 · Depot cold room", "produced_into_cold_room", "picked_to_dock", "DEPOT_COLDROOM", "Monitored"),
        ("2 · Loading dock", "picked_to_dock", "loaded_on_vehicle", "DOCK", "Blind spot"),
        ("3 · Vehicle transit", "loaded_on_vehicle", "unloading_started", veh, "Monitored"),
        ("4 · Unloading at store", "unloading_started", "received_at_store", veh, "Monitored"),
        ("5 · Store floor", "received_at_store", "placed_in_display", f"{store}_FLOOR", "Blind spot"),
        ("6 · Display cabinet", "placed_in_display", None, store, "Monitored"),
    ]
    rows = []
    for name, a, b, loc, cov in defs:
        t0 = T.get(a)
        t1 = T.get(b) if b else end_time
        if t0 is None or t1 is None or pd.isna(t0) or pd.isna(t1) or t1 < t0:
            continue
        hours = (t1 - t0).total_seconds() / 3600
        rows.append({"stage": name, "location": loc, "start": t0, "end": t1,
                     "duration_h": hours, "duration": fmt_dur(hours), "coverage": cov})
    return pd.DataFrame(rows)


def collect_temperatures(sensors, stages: pd.DataFrame) -> pd.DataFrame:
    """
    Temperatures only for the monitored stages. Vehicle readings are limited to
    loaded_on_vehicle -> received_at_store (stage 3 + 4), so a parked vehicle is never included.
    """
    frames = []
    for _, s in stages[stages["coverage"] == "Monitored"].iterrows():
        r = slice_sensor(sensors, s["location"], s["start"], s["end"])
        if r.empty:
            continue
        r["stage"] = s["stage"]
        frames.append(r)
    if not frames:
        return pd.DataFrame()
    t = pd.concat(frames).drop_duplicates(subset=["sensor_id", "timestamp"], keep="first")
    t = t.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    # replace isolated sensor spikes by the rolling median
    med = t.groupby("sensor_id", observed=True)["temperature_c"].transform(
        lambda s: s.rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()
    )
    t["is_spike"] = (t["temperature_c"] - med).abs() >= SPIKE_DELTA
    t["temp_used"] = np.where(t["is_spike"], med, t["temperature_c"])
    return t


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def gantt_figure(stages: pd.DataFrame) -> go.Figure:
    fig = px.timeline(
        stages, x_start="start", x_end="end", y="stage", color="coverage",
        color_discrete_map=COVERAGE_COLORS, text="duration",
        category_orders={"stage": list(stages["stage"])},
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(textposition="auto", marker_line_width=0)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Sensor coverage")
    return fig


def temperature_figure(tdf: pd.DataFrame, stages: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for (stage, sid), g in tdf.groupby(["stage", "sensor_id"], sort=False, observed=True):
        fig.add_trace(go.Scatter(
            x=g["timestamp"], y=g["temp_used"], mode="lines", name=f"{sid} · {stage[4:]}",
            line=dict(color=STAGE_COLORS.get(stage, "#444"), width=1.8,
                      dash="dot" if str(sid).endswith("2") else "solid"),
        ))
    # blind spots as grey bands
    for _, s in stages[stages["coverage"] == "Blind spot"].iterrows():
        fig.add_shape(type="rect", x0=s["start"], x1=s["end"], y0=0, y1=1, xref="x", yref="paper",
                      fillcolor="rgba(150,150,150,0.25)", line=dict(width=0), layer="below")
        fig.add_annotation(x=s["start"] + (s["end"] - s["start"]) / 2, y=1.0, xref="x", yref="paper",
                           text=f"BLIND: {s['stage'][4:]}<br>{s['duration']}", showarrow=False,
                           font=dict(size=10, color="#888"), yanchor="bottom")
    fig.add_hline(y=4, line_dash="dash", line_color="#2e7d32", annotation_text="4 °C safe limit",
                  annotation_position="bottom right")
    fig.add_hline(y=8, line_dash="dash", line_color="#c62828", annotation_text="8 °C danger limit",
                  annotation_position="top right")
    fig.update_layout(height=450, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
                      yaxis_title="Temperature (°C)", legend=dict(orientation="h", y=-0.15))
    return fig


# --------------------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------------------
def compute_alerts(T: dict, ml_su: pd.DataFrame):
    alerts = []  # (level, text)

    floor_h = None
    if T.get("received_at_store") is not None and T.get("placed_in_display") is not None:
        floor_h = (T["placed_in_display"] - T["received_at_store"]).total_seconds() / 3600
    if floor_h is not None and floor_h > FLOOR_WARN_H:
        alerts.append(("warn", f"Store-floor wait of **{fmt_dur(floor_h)}** (no sensor) exceeds 30 min."))

    if not ml_su.empty:
        if (ml_su["meas_dh_above_8"].fillna(0) > 0).any():
            alerts.append(("warn", f"Temperature above 8 °C recorded (`meas_dh_above_8` = "
                                   f"**{ml_su['meas_dh_above_8'].max():.2f} °C·h**)."))
        arr = ml_su[ml_su["checkpoint"] == "arrival"]
        if not arr.empty and arr["target_rsl_days"].iloc[0] < MIN_RSL_DELIVERY:
            alerts.append(("red", f"RSL at arrival is **{arr['target_rsl_days'].iloc[0]:.2f} d** (< 2 d)."))
        over = ml_su[ml_su["target_over_limit"]]
        if not over.empty:
            alerts.append(("red", "Spoilage limit exceeded at: " + ", ".join(f"`{c}`" for c in over["checkpoint"]) + "."))
    return alerts


# --------------------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------------------
def main():
    if not DATA_DIR.exists():
        st.error(f"Data folder not found: `{DATA_DIR}`")
        st.stop()

    lots, su, ev, ml, sites = load_tables()
    sensors = load_sensors()
    idx = build_index(su, ev, ml, sites)

    # ------------------------------- Sidebar --------------------------------------
    with st.sidebar:
        st.title("🥛 Cold-Chain Digital Twin")
        st.caption("ENG499 · CSE-3 · Screen 1: SU tracking")

        mode = st.radio("SU list", ["All SUs", "✅ Normal (no alerts)", "⚠️ With alerts"])

        base = idx[idx["delivery_date"] >= WARMUP_END]            # warm-up week excluded
        if mode.startswith("✅"):
            base = base[base["normal"]]
        elif mode.startswith("⚠️"):
            base = base[base["any_alert"]]
        base = base.sort_values(["delivery_date", "su_id"])
        st.caption(f"{len(base):,} SUs")
        if base.empty:
            st.warning("No SU found.")
            st.stop()

        labels = {r.su_id: f"{r.su_id} | {r.store} | {r.delivery_date:%d %b} {r.flags}" for r in base.itertuples()}
        sel = st.selectbox("Shipment unit (SU)", list(labels), format_func=lambda i: labels[i])
        st.caption("🕒 floor wait > 30 min · 🔥 temp > 8 °C · 🔴 low RSL")

    # --------------------------- Selected SU --------------------------------------
    row = idx[idx["su_id"] == sel].iloc[0]
    hist = get_history(row, ev)
    T = hist.groupby("event")["timestamp"].min().to_dict()
    if "placed_in_display" not in T and pd.notna(row["placed_in_display"]):
        T["placed_in_display"] = row["placed_in_display"]

    if T.get("discarded_expired") is not None:
        end_time, outcome = T["discarded_expired"], "Discarded at label expiry"
    elif pd.notna(row["last_unit_gone"]):
        end_time, outcome = row["last_unit_gone"], "Sold out"
    else:
        end_time, outcome = None, "Unknown"

    stages = build_stages(T, row, end_time)
    tdf = collect_temperatures(sensors, stages) if not stages.empty else pd.DataFrame()
    ml_su = ml[ml["su_id"] == sel].copy()
    ml_su["_ord"] = ml_su["checkpoint"].map({c: i for i, c in enumerate(CHECKPOINT_ORDER)})
    ml_su = ml_su.sort_values(["_ord", "checkpoint_time"]).drop(columns="_ord")

    # ------------------------------- Header ---------------------------------------
    st.title("Digital Twin · Shipment Unit Tracker")
    lot_info = lots[lots["lot_id"] == row["lot_id"]]
    lot_txt = ""
    if not lot_info.empty:
        lot_txt = (f" · lot produced {pd.Timestamp(lot_info['production_time'].iloc[0]):%d %b %Y %H:%M}"
                   f", label expiry {lot_info['label_expiry_date'].iloc[0]}")
    st.caption(f"Outcome: **{outcome}**{lot_txt}")

    cols = st.columns(6)
    cols[0].metric("SU", row["su_id"])
    cols[1].metric("Lot", row["lot_id"])
    cols[2].metric("Store (cabinet)", f"{row['store']} ({row['cabinet_class']})")
    cols[3].metric("Vehicle / route", f"{row['vehicle_id']} / {row['route']}")
    cols[4].metric("Crates", int(row["crates"]))
    cols[5].metric("Delivery date", f"{row['delivery_date']:%d %b %Y}")

    if stages.empty:
        st.error("Could not reconstruct the journey of this SU (missing events).")
        st.stop()

    # ------------------------- 1 · Where & when -----------------------------------
    st.header("1 · Where it was and when")
    st.plotly_chart(gantt_figure(stages), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Stage durations**")
        st.dataframe(
            stages.assign(start=stages["start"].dt.strftime("%d %b %H:%M"),
                          end=stages["end"].dt.strftime("%d %b %H:%M"))
            [["stage", "location", "start", "end", "duration", "coverage"]],
            hide_index=True, width="stretch",
        )
        blind_h = stages.loc[stages["coverage"] == "Blind spot", "duration_h"].sum()
        st.caption(f"Time without sensor coverage (dock + store floor): {fmt_dur(blind_h)}")
    with right:
        st.markdown("**Event log**")
        evt = hist.copy()
        evt["timestamp"] = evt["timestamp"].dt.strftime("%d %b %H:%M")
        st.dataframe(evt[["timestamp", "event", "location", "qty"]], hide_index=True, width="stretch")

    # ------------------------- 2 · Temperature ------------------------------------
    st.header("2 · Temperature history")
    if tdf.empty:
        st.warning("No sensor readings found for the monitored stages of this SU.")
    else:
        st.plotly_chart(temperature_figure(tdf, stages), width="stretch")
        st.caption("Vehicle readings are limited to the time the milk was on the vehicle. "
                   "Grey bands are blind spots (no sensor). Isolated sensor spikes are replaced by a rolling median.")

    # ------------------------- 3 · RSL & alerts -----------------------------------
    st.header("3 · Remaining shelf life (RSL) & alerts")
    alerts = compute_alerts(T, ml_su)
    if not alerts:
        st.success("No alerts for this SU.")
    for level, text in alerts:
        (st.error if level == "red" else st.warning)(("🔴 " if level == "red" else "⚠️ ") + text)

    if ml_su.empty:
        st.info("This SU has no rows in rsl_training_table.csv.")
        return

    arr = ml_su[ml_su["checkpoint"] == "arrival"]
    r = arr.iloc[0] if not arr.empty else ml_su.iloc[0]
    m = st.columns(3)
    m[0].metric("RSL at arrival (target_rsl_days)", f"{r['target_rsl_days']:.2f} d")
    m[1].metric("Label days left", f"{r['label_days_left']:.2f} d")
    m[2].metric("Equivalent age @ 4 °C", f"{r['equiv_age_at_4c_days']:.2f} d")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ml_su["checkpoint"], y=ml_su["target_rsl_days"], mode="lines+markers",
                             name="Sensor-based RSL (days)", line=dict(color="#c62828", width=3)))
    fig.add_trace(go.Scatter(x=ml_su["checkpoint"], y=ml_su["label_days_left"], mode="lines+markers",
                             name="Label days left", line=dict(color="#1f77b4", dash="dash")))
    fig.add_hline(y=MIN_RSL_DELIVERY, line_dash="dot", line_color="#c62828", annotation_text="2-day limit")
    fig.update_layout(height=360, yaxis_title="Days", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")


main()