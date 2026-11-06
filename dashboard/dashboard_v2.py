"""
ENG499 Cold-Chain Digital Twin - Dashboard v1 (Part B, CSE-3)
=============================================================
Extends the Week-2 screen into a first dashboard that talks to the backend API (backend/main.py):
  * Overview            list of shipment units (SUs) with alert flags, alert counts per store
  * Shipment unit       current state at a selectable time (replay): location, last temperature, RSL,
                        alerts, temperature history, journey timeline, RSL checkpoints, event log

Start the backend first (repository root):
    python -m uvicorn backend.main:app --port 8000
Then:
    python -m streamlit run dashboard/dashboard_v2.py
Other API address:  set COLDCHAIN_API=http://host:8000
"""
from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("COLDCHAIN_API", "http://127.0.0.1:8000").rstrip("/")
TEMP_LIMIT_C, TEMP_SAFE_C, MIN_RSL_D = 8.0, 4.0, 2.0

st.set_page_config(page_title="Cold-Chain Digital Twin", page_icon="🥛", layout="wide")
st.markdown("<style>[data-testid='stAppDeployButton'], .stAppDeployButton {display: none;}</style>",
            unsafe_allow_html=True)

STATUS_ICON = {"ok": "🟢", "warning": "🟠", "critical": "🔴"}
STATUS_COLOR = {"ok": "#2e7d32", "warning": "#ef6c00", "critical": "#c62828"}
STAGE_COLORS = {"1 · Depot cold room": "#1f77b4", "3 · Vehicle transit": "#2ca02c",
                "4 · Unloading at store": "#ff7f0e", "6 · Display cabinet": "#9467bd"}
COVERAGE_COLORS = {"Monitored": "#2e7d32", "Blind spot": "#c62828"}


# --------------------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------------------
class ApiError(Exception):
    pass


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, params: tuple = ()):
    try:
        r = requests.get(f"{API}{path}", params=dict(params), timeout=60)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the backend at {API}") from exc
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(f"{r.status_code}: {detail}")
    return r.json()


def get(path: str, **params):
    return api_get(path, tuple(sorted((k, v) for k, v in params.items() if v is not None)))


def fmt_ts(iso: str | None, with_year: bool = False) -> str:
    return "n/a" if not iso else pd.Timestamp(iso).strftime("%d %b %Y %H:%M" if with_year else "%d %b %H:%M")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _vline(fig: go.Figure, ts: pd.Timestamp, label: str) -> None:
    x = ts.to_pydatetime()
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="#111", width=2, dash="solid"))
    fig.add_annotation(x=x, y=1.0, xref="x", yref="paper", text=label, showarrow=False,
                       yanchor="bottom", font=dict(size=11, color="#111"))


def temperature_figure(temp: dict, journey: dict, as_of: pd.Timestamp) -> go.Figure:
    fig = go.Figure()
    s = pd.DataFrame(temp["series"])
    s["timestamp"] = pd.to_datetime(s["timestamp"])
    for (stage, sid), g in s.groupby(["stage", "sensor_id"], sort=False):
        fig.add_trace(go.Scatter(
            x=g["timestamp"], y=g["temperature_c"], mode="lines", name=f"{sid} · {stage[4:]}",
            line=dict(color=STAGE_COLORS.get(stage, "#444"), width=1.8,
                      dash="dot" if str(sid).endswith("2") else "solid"),
        ))
    hot = s[s["temperature_c"] > TEMP_LIMIT_C]
    if not hot.empty:
        fig.add_trace(go.Scatter(x=hot["timestamp"], y=hot["temperature_c"], mode="markers",
                                 name=f"> {TEMP_LIMIT_C:g} °C", marker=dict(color="#c62828", size=6)))
    for st_ in journey["stages"]:
        if st_["coverage"] == "Blind spot":
            x0, x1 = pd.Timestamp(st_["start"]).to_pydatetime(), pd.Timestamp(st_["end"]).to_pydatetime()
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, xref="x", yref="paper",
                          fillcolor="rgba(150,150,150,0.25)", line=dict(width=0), layer="below")
            fig.add_annotation(x=x0 + (x1 - x0) / 2, y=1.0, xref="x", yref="paper", yanchor="bottom",
                               text=f"BLIND: {st_['stage'][4:]}<br>{st_['duration']}", showarrow=False,
                               font=dict(size=10, color="#888"))
    end = pd.Timestamp(journey["end"]).to_pydatetime()
    if as_of < pd.Timestamp(journey["end"]):          # "future" part of the replay
        fig.add_shape(type="rect", x0=as_of.to_pydatetime(), x1=end, y0=0, y1=1, xref="x", yref="paper",
                      fillcolor="rgba(255,255,255,0.65)", line=dict(width=0), layer="above")
    _vline(fig, as_of, "now")
    fig.add_hline(y=TEMP_SAFE_C, line_dash="dash", line_color="#2e7d32", annotation_text=f"{TEMP_SAFE_C:g} °C safe",
                  annotation_position="bottom right")
    fig.add_hline(y=TEMP_LIMIT_C, line_dash="dash", line_color="#c62828",
                  annotation_text=f"{TEMP_LIMIT_C:g} °C alert limit", annotation_position="top right")
    fig.update_layout(height=430, hovermode="x unified", margin=dict(l=10, r=10, t=70, b=10),
                      yaxis_title="Temperature (°C)", legend=dict(orientation="h", y=-0.15))
    return fig


def gantt_figure(journey: dict, as_of: pd.Timestamp) -> go.Figure:
    d = pd.DataFrame(journey["stages"])
    d["start"], d["end"] = pd.to_datetime(d["start"]), pd.to_datetime(d["end"])
    fig = px.timeline(d, x_start="start", x_end="end", y="stage", color="coverage",
                      color_discrete_map=COVERAGE_COLORS, text="duration",
                      category_orders={"stage": list(d["stage"])})
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(textposition="auto", marker_line_width=0)
    _vline(fig, as_of, "now")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="Sensor coverage")
    return fig


def rsl_figure(rsl: dict) -> go.Figure:
    d = pd.DataFrame(rsl["checkpoints"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["checkpoint"], y=d["predicted_days"], mode="lines+markers", name="RSL (days)",
                             line=dict(color="#c62828", width=3)))
    fig.add_trace(go.Scatter(x=d["checkpoint"], y=d["label_days_left"], mode="lines+markers",
                             name="Label days left", line=dict(color="#1f77b4", dash="dash")))
    cur = rsl.get("current")
    if cur:
        fig.add_trace(go.Scatter(x=[cur["checkpoint"]], y=[cur["predicted_days"]], mode="markers", name="valid now",
                                 marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="#111", width=3))))
    fig.add_hline(y=MIN_RSL_D, line_dash="dot", line_color="#c62828", annotation_text="2-day minimum")
    fig.update_layout(height=340, yaxis_title="Days", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


# --------------------------------------------------------------------------------------
# Page parts
# --------------------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.title("🥛 Cold-Chain Digital Twin")
        st.caption("ENG499 · CSE-3 · Dashboard v1")
        try:
            h = get("/health")
            st.success(f"Backend OK · {h['sensor_source']} · {h['sensor_rows']:,} readings", icon="✅")
            mq = h.get("mqtt", {})
            if mq.get("enabled"):
                st.caption(f"MQTT: {'connected' if mq.get('connected') else 'not connected'} · "
                           f"{mq.get('messages', 0):,} messages")
            stores = get("/stores")
        except ApiError as exc:
            st.error(f"{exc}\n\nStart it with:\n`python -m uvicorn backend.main:app --port 8000`")
            st.stop()

        mode = st.radio("SU list", ["All SUs", "✅ Normal (no alerts)", "⚠️ With alerts"])
        status = "normal" if mode.startswith("✅") else ("alert" if mode.startswith("⚠️") else "all")
        store = st.selectbox("Store", ["All stores"] + stores)
        units = get("/shipment-units", status=status, store=None if store == "All stores" else store)
        st.caption(f"{len(units):,} SUs")
        if not units:
            st.warning("No SU found.")
            st.stop()
        labels = {u["su_id"]: f"{STATUS_ICON[u['status']]} {u['su_id']} | {u['store']} | {u['delivery_date']}"
                  for u in units}
        sel = st.selectbox("Shipment unit (SU)", list(labels), format_func=lambda i: labels[i])
        st.caption("🟢 no alert · 🟠 warning · 🔴 critical (low RSL / limit exceeded)")
    return units, sel


def overview_tab(units: list[dict]):
    d = pd.DataFrame(units)
    c = st.columns(4)
    c[0].metric("Shipment units", len(d))
    c[1].metric("🟢 No alert", int((d["status"] == "ok").sum()))
    c[2].metric("🟠 Warning", int((d["status"] == "warning").sum()))
    c[3].metric("🔴 Critical", int((d["status"] == "critical").sum()))

    table = pd.DataFrame({
        "": d["status"].map(STATUS_ICON), "SU": d["su_id"], "Lot": d["lot_id"], "Store": d["store"],
        "Delivered": d["delivery_date"], "Vehicle": d["vehicle_id"], "Crates": d["crates"],
        "RSL at arrival (d)": d["rsl_at_arrival_days"].round(2),
        "Floor wait (h)": d["floor_h"].round(2),
        "🕒": d["floor_alert"].map({True: "🕒", False: ""}),
        "🔥": d["heat_alert"].map({True: "🔥", False: ""}),
        "🔴": d["red_alert"].map({True: "🔴", False: ""}),
    })
    st.dataframe(table, hide_index=True, width="stretch", height=380)
    st.caption("🕒 store-floor wait > 30 min · 🔥 temperature above 8 °C recorded · 🔴 RSL at arrival < 2 d or limit exceeded")

    by_store = d.groupby(["store", "status"]).size().reset_index(name="SUs")
    fig = px.bar(by_store, x="store", y="SUs", color="status", color_discrete_map=STATUS_COLOR,
                 category_orders={"status": ["ok", "warning", "critical"]})
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=None, legend_title_text="status")
    st.markdown("**Alerts per store**")
    st.plotly_chart(fig, width="stretch")


def _jump(key: str, ts):
    st.session_state[key] = ts


def su_tab(su_id: str):
    try:
        jr = get(f"/shipment-units/{su_id}/journey")
    except ApiError as exc:
        st.error(str(exc))
        return

    start, end = pd.Timestamp(jr["start"]).to_pydatetime(), pd.Timestamp(jr["end"]).to_pydatetime()
    lot = jr.get("lot", {})
    st.subheader(f"{su_id} · {jr['store']} ({jr.get('cabinet_class')})")
    st.caption(f"Outcome: **{jr['outcome']}**"
               + (f" · lot produced {fmt_ts(lot.get('production_time'), True)}, label expiry {lot.get('label_expiry_date')}"
                  if lot else "")
               + f" · vehicle {jr['vehicle_id']} / route {jr['route']} · {jr['crates']} crates")

    # ---- time control (replay) ----------------------------------------------------------
    key = f"asof_{su_id}"
    if key not in st.session_state:
        st.session_state[key] = end
    st.markdown("**Time (replay of the journey)**")
    bcols = st.columns(len(jr["stages"]) + 1)
    for col, s in zip(bcols, jr["stages"]):
        col.button(s["stage"][4:], key=f"{key}_{s['stage']}", width="stretch", on_click=_jump,
                   args=(key, pd.Timestamp(s["start"]).to_pydatetime()))
    bcols[-1].button("End", key=f"{key}_end", width="stretch", on_click=_jump, args=(key, end))
    as_of_dt = st.slider("Time", min_value=start, max_value=end, step=timedelta(minutes=15),
                         format="DD MMM HH:mm", key=key, label_visibility="collapsed")
    as_of_iso = pd.Timestamp(as_of_dt).isoformat()

    try:
        state = get(f"/shipment-units/{su_id}/state", as_of=as_of_iso)
        temp = get(f"/shipment-units/{su_id}/temperature")
        rsl = get(f"/shipment-units/{su_id}/rsl", as_of=as_of_iso)
    except ApiError as exc:
        st.error(str(exc))
        return
    as_of = pd.Timestamp(state["as_of"])

    # ---- current state ---------------------------------------------------------------------
    st.header("Current state")
    banner = {"ok": st.success, "warning": st.warning, "critical": st.error}[state["status"]]
    banner(f"{STATUS_ICON[state['status']]} Status at {fmt_ts(state['as_of'])}: **{state['status'].upper()}**"
           + ("" if state["alerts"] else " · no alerts"))

    loc, lt, rs = state["location"], state["last_temperature"], state["rsl"]
    m = st.columns(4)
    if loc["stage"]:
        m[0].metric("Location", loc["stage"][4:], f"{loc['asset_id']} · {loc['time_in_stage']} here", delta_color="off")
    else:
        m[0].metric("Location", "Not started")
    if lt:
        stale = " · stale" if lt["stale"] else ""
        m[1].metric("Last temperature", f"{lt['value_c']:.1f} °C",
                    f"{lt['sensor_id']} · {lt['age_minutes']:.0f} min ago{stale}",
                    delta_color="inverse" if lt["above_limit"] else "off")
    else:
        m[1].metric("Last temperature", "n/a", "no reading yet", delta_color="off")
    if rs and rs["predicted_days"] is not None:
        extra = f"label: {rs['label_days_left']:.1f} d left · {rs['checkpoint']}" if rs["label_days_left"] is not None else rs["checkpoint"]
        m[2].metric("RSL", f"{rs['predicted_days']:.2f} d", extra, delta_color="off")
    else:
        m[2].metric("RSL", "n/a", "available from arrival at the store", delta_color="off")
    m[3].metric("Peak temperature so far", "n/a" if state["max_temperature_c"] is None else f"{state['max_temperature_c']:.1f} °C",
                f"{state['minutes_above_limit']} min above {TEMP_LIMIT_C:g} °C", delta_color="off")

    for a in state["alerts"]:
        (st.error if a["level"] == "critical" else st.warning)(
            ("🔴 " if a["level"] == "critical" else "⚠️ ") + a["message"])
    if rs and rs.get("source"):
        st.caption(f"RSL source: {rs['source']}" + (f" · uncertainty ± {rs['uncertainty_days']:.2f} d" if rs.get("uncertainty_days") else ""))

    # ---- journey + temperature -------------------------------------------------------------
    st.header("Where it was and when")
    st.plotly_chart(gantt_figure(jr, as_of), width="stretch")
    st.caption(f"Time without sensor coverage (dock + store floor): {jr['blind_spot_hours']:.2f} h")

    st.header("Temperature history")
    if not temp["series"]:
        st.warning("No sensor readings found for the monitored stages of this SU.")
    else:
        st.plotly_chart(temperature_figure(temp, jr, as_of), width="stretch")
        st.caption("Vehicle readings only while the milk was on board. Grey bands = blind spots. "
                   "Isolated sensor spikes are replaced by a rolling median. TS-U07 drift is corrected.")

    st.header("Remaining shelf life (RSL)")
    if rsl["checkpoints"]:
        st.plotly_chart(rsl_figure(rsl), width="stretch")
    else:
        st.info("This SU has no checkpoints in rsl_training_table.csv.")

    with st.expander("Stage table and event log"):
        l, r = st.columns([3, 2])
        stg = pd.DataFrame(jr["stages"])
        stg["start"] = pd.to_datetime(stg["start"]).dt.strftime("%d %b %H:%M")
        stg["end"] = pd.to_datetime(stg["end"]).dt.strftime("%d %b %H:%M")
        l.dataframe(stg[["stage", "location", "start", "end", "duration", "coverage"]], hide_index=True, width="stretch")
        ev = pd.DataFrame(jr["events"])
        ev["timestamp"] = pd.to_datetime(ev["timestamp"]).dt.strftime("%d %b %H:%M")
        r.dataframe(ev, hide_index=True, width="stretch")


def main():
    units, sel = sidebar()
    st.title("Digital Twin · Cold-Chain Dashboard")
    t1, t2 = st.tabs(["📋 Overview", "🔎 Shipment unit"])
    with t1:
        overview_tab(units)
    with t2:
        su_tab(sel)


main()
