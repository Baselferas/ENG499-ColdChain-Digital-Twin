"""
CSE-1 Incident Detection & MQTT Alert Publisher (Weeks 5 & 6).

Detects 5 cold-chain incident types in the cleaned sensor data and publishes
JSON alerts to MQTT topic: coldchain/alerts/<asset_id>

Incident types detected:
  1. refrigeration_failure         - Depot cold room (both sensors > 7.0 C for >= 30 min) or
                                     vehicle reefer mid-trip (> 10.0 C for >= 30 min after 30 min pre-cool window)
  2. van_loaded_without_precooling - Vehicle loaded (on_board=True) while cargo box > 8.0 C for >= 15 min
  3. door_left_open                - Rapid rise >= 2.0 C in 15 min (> 7.5 C cabinet / > 5.5 C cold room)
                                     that recovers within 15..90 min
  4. cabinet_failure               - Store cabinet > 9.5 C for >= 60 min (or > 8.5 C for >= 120 min)
  5. temp_above_8c                 - Any active cold-chain sensor > 8.0 C for >= 15 min (on_board only for vans)

Usage:
  # 1. Scan the full 6-month cleaned file and generate the incident report CSV:
  python incident_detector.py --batch

  # 2. Run as a live MQTT subscriber that detects incidents in real time and publishes alerts:
  python incident_detector.py --live
"""
import argparse
from collections import defaultdict, deque
import json
import os
import time

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

import message_format as mf

OUT_INCIDENTS_CSV = os.path.join("output_clean", "detected_incidents.csv")
OUT_SUMMARY_MD = os.path.join("output_clean", "incident_summary.md")
ALERT_TOPIC_FMT = "coldchain/alerts/%s"


def make_client(client_id):
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


def make_alert_payload(idx, inc_type, severity, asset_id, sensor_id, placement,
                       start_ts, end_ts, duration_min, peak_temp, message):
    day_tag = str(start_ts)[:10].replace("-", "")
    return {
        "alert_id": "ALT-%s-%s-%04d" % (day_tag, asset_id, idx),
        "incident_type": inc_type,
        "severity": severity,
        "asset_id": asset_id,
        "sensor_id": sensor_id,
        "placement": placement,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "duration_min": int(duration_min),
        "peak_temperature_c": round(float(peak_temp), 2) if peak_temp is not None else None,
        "message": message,
    }


def _extract_runs(mask, ts_series, temp_series, min_slots):
    """Find contiguous True runs in boolean mask lasting at least min_slots (5 min each)."""
    runs = []
    in_run = False
    start_i = 0
    n = len(mask)
    for i in range(n):
        if mask[i] and not in_run:
            in_run = True
            start_i = i
        elif not mask[i] and in_run:
            in_run = False
            length = i - start_i
            if length >= min_slots:
                runs.append((start_i, i - 1))
    if in_run:
        length = n - start_i
        if length >= min_slots:
            runs.append((start_i, n - 1))
    out = []
    for s_i, e_i in runs:
        dur = (e_i - s_i + 1) * 5
        peak = float(np.nanmax(temp_series[s_i:e_i + 1]))
        out.append((ts_series[s_i], ts_series[e_i], dur, peak))
    return out


# ---------------------------------------------------------------------------
# BATCH DETECTION ON FULL CLEANED DATASET
# ---------------------------------------------------------------------------
def detect_all_batch(df):
    """Run all 5 rules across the cleaned DataFrame and return a DataFrame of incidents."""
    df = df.copy()
    df["dtime"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "dtime"]).reset_index(drop=True)
    df["iso_ts"] = df["dtime"].dt.strftime("%Y-%m-%dT%H:%M:%S") + mf.TZ_SUFFIX
    df["on_board_bool"] = df["on_board"].map(mf._bool_or_none)

    incidents = []

    # --- Per-sensor rules ---
    for sid, g in df.groupby("sensor_id", sort=True):
        g = g.reset_index(drop=True)
        asset_id = g["asset_id"].iloc[0]
        placement = g["placement"].iloc[0]
        ts = g["iso_ts"].to_numpy()
        temp = g["temperature_c"].to_numpy(dtype=float)
        valid = ~np.isnan(temp)

        if placement == "vehicle_reefer":
            onb = g["on_board_bool"].fillna(False).to_numpy(dtype=bool)
            active = valid & onb
        else:
            onb = np.ones(len(g), dtype=bool)
            active = valid

        # 1. temp_above_8c: > 8.0 C for >= 15 min (3 slots) while active
        m_high8 = active & (temp > 8.0)
        for s_ts, e_ts, dur, peak in _extract_runs(m_high8, ts, temp, min_slots=3):
            incidents.append({
                "asset_id": asset_id,
                "sensor_id": sid,
                "placement": placement,
                "type": "temp_above_8c",
                "severity": "warning" if peak <= 10.0 else "critical",
                "start": s_ts,
                "end": e_ts,
                "duration_min": dur,
                "peak_temp_c": round(peak, 2),
                "details": "Temperature exceeded 8.0 C for %d min (peak %.2f C)" % (dur, peak),
            })

        # 2. door_left_open: display_cabinet or cold_room_front rapid rise + recovery in 15..90 min
        if placement in ("display_cabinet", "cold_room_front"):
            thresh = 7.5 if placement == "display_cabinet" else 5.5
            recov = 6.5 if placement == "display_cabinet" else 5.0
            # Baseline: rolling median of previous 30 min (6 slots) shifted by 1
            s_temp = pd.Series(temp)
            base = s_temp.shift(1).rolling(6, min_periods=2).median().to_numpy()
            excursion = valid & (temp > thresh) & ((temp - base) >= 1.5)
            # Expand excursion to contiguous slots where temp stays above recovery threshold
            above_rec = valid & (temp > recov)
            for s_i, e_i in _find_candidate_blocks(above_rec):
                dur = (e_i - s_i + 1) * 5
                # Must contain a sharp rise (excursion) and recover within 15..90 min
                if 15 <= dur <= 90 and np.any(excursion[s_i:e_i + 1]):
                    peak = float(np.nanmax(temp[s_i:e_i + 1]))
                    incidents.append({
                        "asset_id": asset_id,
                        "sensor_id": sid,
                        "placement": placement,
                        "type": "door_left_open",
                        "severity": "warning",
                        "start": ts[s_i],
                        "end": ts[e_i],
                        "duration_min": dur,
                        "peak_temp_c": round(peak, 2),
                        "details": "Door left open: rapid warming to %.2f C, recovered after %d min" % (peak, dur),
                    })

        # 3. van_loaded_without_precooling: vehicle trip starts with temp > 8.0 C for >= 15 min
        if placement == "vehicle_reefer":
            trip_start = onb & (~np.roll(onb, 1))
            trip_start[0] = onb[0]
            for idx_start in np.where(trip_start)[0]:
                # Check first 6 slots (30 min) of trip
                end_win = min(idx_start + 6, len(g))
                seg_t = temp[idx_start:end_win]
                seg_onb = onb[idx_start:end_win]
                hot_slots = (~np.isnan(seg_t)) & seg_onb & (seg_t > 8.0)
                if np.sum(hot_slots) >= 3 and (not np.isnan(temp[idx_start])) and temp[idx_start] > 8.0:
                    # Find when it finally cools <= 8.0 C or trip ends
                    k = idx_start
                    while k < len(g) and onb[k] and (np.isnan(temp[k]) or temp[k] > 8.0):
                        k += 1
                    end_i = max(idx_start, k - 1)
                    dur = (end_i - idx_start + 1) * 5
                    peak = float(np.nanmax(temp[idx_start:end_i + 1]))
                    incidents.append({
                        "asset_id": asset_id,
                        "sensor_id": sid,
                        "placement": placement,
                        "type": "van_loaded_without_precooling",
                        "severity": "critical",
                        "start": ts[idx_start],
                        "end": ts[end_i],
                        "duration_min": dur,
                        "peak_temp_c": round(peak, 2),
                        "details": "Vehicle %s loaded without pre-cooling (start %.2f C, >8 C for %d min)" % (
                            asset_id, temp[idx_start], dur),
                    })

        # 4. cabinet_failure: display_cabinet > 9.5 C for >= 60 min (12 slots)
        #    OR > 8.5 C for >= 120 min (24 slots)
        if placement == "display_cabinet":
            m_cab_crit = valid & (temp > 9.5)
            crit_runs = _extract_runs(m_cab_crit, ts, temp, min_slots=12)
            for s_ts, e_ts, dur, peak in crit_runs:
                incidents.append({
                    "asset_id": asset_id,
                    "sensor_id": sid,
                    "placement": placement,
                    "type": "cabinet_failure",
                    "severity": "critical",
                    "start": s_ts,
                    "end": e_ts,
                    "duration_min": dur,
                    "peak_temp_c": round(peak, 2),
                    "details": "Display cabinet failure: sustained >9.5 C for %d min (peak %.2f C)" % (dur, peak),
                })
            # Also check prolonged > 8.5 C for >= 120 min if not already covered
            m_cab_warn = valid & (temp > 8.5)
            for s_ts, e_ts, dur, peak in _extract_runs(m_cab_warn, ts, temp, min_slots=24):
                if not any(s_ts <= cs and ce <= e_ts for cs, ce, _, _ in crit_runs):
                    note = " (note: TS-U07 sensor drift)" if sid == "TS-U07" else ""
                    incidents.append({
                        "asset_id": asset_id,
                        "sensor_id": sid,
                        "placement": placement,
                        "type": "cabinet_failure",
                        "severity": "critical",
                        "start": s_ts,
                        "end": e_ts,
                        "duration_min": dur,
                        "peak_temp_c": round(peak, 2),
                        "details": "Display cabinet failure: sustained >8.5 C for %d min (peak %.2f C)%s" % (
                            dur, peak, note),
                    })

        # 5a. refrigeration_failure (Vehicles mid-trip: > 10.0 C for >= 30 min after first 30 min of trip)
        if placement == "vehicle_reefer":
            # Build mask for slots at least 30 min (6 slots) into an active trip
            mid_trip = np.zeros(len(g), dtype=bool)
            run_onb = 0
            for i in range(len(g)):
                if onb[i]:
                    run_onb += 1
                    if run_onb > 6:
                        mid_trip[i] = True
                else:
                    run_onb = 0
            m_veh_fail = valid & mid_trip & (temp > 10.0)
            for s_ts, e_ts, dur, peak in _extract_runs(m_veh_fail, ts, temp, min_slots=6):
                incidents.append({
                    "asset_id": asset_id,
                    "sensor_id": sid,
                    "placement": placement,
                    "type": "refrigeration_failure",
                    "severity": "critical",
                    "start": s_ts,
                    "end": e_ts,
                    "duration_min": dur,
                    "peak_temp_c": round(peak, 2),
                    "details": "Vehicle %s reefer failure in transit: >10.0 C for %d min (peak %.2f C)" % (
                        asset_id, dur, peak),
                })

    # 5b. refrigeration_failure (Depot Cold Room: BOTH TS-CR-1 and TS-CR-2 > 7.0 C for >= 30 min)
    cr = df[df["asset_id"] == "DEPOT_COLDROOM"].pivot(index="iso_ts", columns="sensor_id", values="temperature_c")
    if "TS-CR-1" in cr.columns and "TS-CR-2" in cr.columns:
        cr = cr.sort_index()
        both_hot = (cr["TS-CR-1"] > 7.0) & (cr["TS-CR-2"] > 7.0)
        cr_max = cr[["TS-CR-1", "TS-CR-2"]].max(axis=1).to_numpy()
        for s_ts, e_ts, dur, peak in _extract_runs(both_hot.to_numpy(), cr.index.to_numpy(), cr_max, min_slots=6):
            incidents.append({
                "asset_id": "DEPOT_COLDROOM",
                "sensor_id": "TS-CR-1+TS-CR-2",
                "placement": "cold_room",
                "type": "refrigeration_failure",
                "severity": "critical",
                "start": s_ts,
                "end": e_ts,
                "duration_min": dur,
                "peak_temp_c": round(peak, 2),
                "details": "Depot cold room refrigeration failure: both sensors >7.0 C for %d min (peak %.2f C)" % (
                    dur, peak),
            })

    out_df = pd.DataFrame(incidents)
    if not out_df.empty:
        out_df = out_df.sort_values(["start", "asset_id", "type"]).reset_index(drop=True)
    return out_df


def _find_candidate_blocks(mask):
    blocks = []
    in_b = False
    s = 0
    for i, val in enumerate(mask):
        if val and not in_b:
            in_b = True
            s = i
        elif not val and in_b:
            in_b = False
            blocks.append((s, i - 1))
    if in_b:
        blocks.append((s, len(mask) - 1))
    return blocks


# ---------------------------------------------------------------------------
# LIVE MQTT STREAM DETECTOR
# ---------------------------------------------------------------------------
class LiveIncidentMonitor:
    """Stateful sliding-window detector that subscribes to telemetry and publishes alerts."""

    def __init__(self, client):
        self.client = client
        self.history = defaultdict(lambda: deque(maxlen=36))  # last 3 hours (36 * 5 min)
        self.active_alerts = {}  # (sensor_id, incident_type) -> start_timestamp
        self.alert_counter = 0

    def publish_alert(self, inc_type, severity, asset_id, sensor_id, placement,
                      start_ts, end_ts, duration_min, peak_temp, message):
        self.alert_counter += 1
        payload = make_alert_payload(
            self.alert_counter, inc_type, severity, asset_id, sensor_id, placement,
            start_ts, end_ts, duration_min, peak_temp, message
        )
        topic = ALERT_TOPIC_FMT % asset_id
        self.client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=1, retain=False)
        print("[ALERT -> %s] %s | %s (%s) | %s -> %s | peak=%.2f C | %s" % (
            topic, inc_type.upper(), asset_id, sensor_id, start_ts, end_ts, peak_temp, message))

    def process_message(self, m):
        sid = m["sensor_id"]
        asset_id = m["asset_id"]
        placement = m["placement"]
        ts = m["timestamp"]
        temp = m["temperature_c"]
        on_board = m.get("on_board")

        self.history[sid].append((ts, temp, on_board))
        hist = list(self.history[sid])

        # Ignore gaps for threshold triggers
        if temp is None:
            return

        is_vehicle = (placement == "vehicle_reefer")
        active_now = bool(on_board) if is_vehicle else True

        # 1. temp_above_8c (>= 3 consecutive slots > 8.0 C while active)
        if active_now and len(hist) >= 3:
            last3 = hist[-3:]
            if all((t is not None and t > 8.0 and (bool(ob) if is_vehicle else True)) for _, t, ob in last3):
                key = (sid, "temp_above_8c")
                if key not in self.active_alerts:
                    self.active_alerts[key] = last3[0][0]
                    peak = max(t for _, t, _ in last3)
                    self.publish_alert(
                        "temp_above_8c", "warning", asset_id, sid, placement,
                        last3[0][0], ts, 15, peak,
                        "%s exceeded 8.0 C for >=15 min (now %.2f C)" % (asset_id, temp)
                    )
            else:
                self.active_alerts.pop((sid, "temp_above_8c"), None)
        else:
            self.active_alerts.pop((sid, "temp_above_8c"), None)

        # 2. van_loaded_without_precooling (first 3 slots of trip all > 8.0 C)
        if is_vehicle and active_now and len(hist) >= 4:
            # Check if trip started 2 slots ago (hist[-4] was False/None, hist[-3..-1] all True and > 8 C)
            if not bool(hist[-4][2]):
                trip3 = hist[-3:]
                if all(bool(ob) and t is not None and t > 8.0 for _, t, ob in trip3):
                    peak = max(t for _, t, _ in trip3)
                    self.publish_alert(
                        "van_loaded_without_precooling", "critical", asset_id, sid, placement,
                        trip3[0][0], ts, 15, peak,
                        "Vehicle %s loaded without pre-cooling (started at %.2f C)" % (asset_id, trip3[0][1])
                    )

        # 3. door_left_open (display_cabinet or cold_room_front: spiked > thresh and just recovered <= recov within 15..90 min)
        if placement in ("display_cabinet", "cold_room_front") and len(hist) >= 6:
            thresh = 7.5 if placement == "display_cabinet" else 5.5
            recov = 6.5 if placement == "display_cabinet" else 5.0
            prev_t = hist[-2][1]
            if temp <= recov and prev_t is not None and prev_t > recov:
                # Walk backwards to find when excursion above recov started
                k = len(hist) - 2
                while k >= 0 and hist[k][1] is not None and hist[k][1] > recov:
                    k -= 1
                start_idx = k + 1
                dur = (len(hist) - 1 - start_idx) * 5
                seg = [x[1] for x in hist[start_idx:-1] if x[1] is not None]
                if 15 <= dur <= 90 and seg and max(seg) > thresh:
                    self.publish_alert(
                        "door_left_open", "warning", asset_id, sid, placement,
                        hist[start_idx][0], ts, dur, max(seg),
                        "%s door left open: rose to %.2f C and recovered in %d min" % (asset_id, max(seg), dur)
                    )

        # 4. cabinet_failure (display_cabinet > 9.5 C for >= 60 min = 12 slots)
        if placement == "display_cabinet" and len(hist) >= 12:
            last12 = hist[-12:]
            if all(t is not None and t > 9.5 for _, t, _ in last12):
                key = (sid, "cabinet_failure")
                if key not in self.active_alerts:
                    self.active_alerts[key] = last12[0][0]
                    peak = max(t for _, t, _ in last12)
                    self.publish_alert(
                        "cabinet_failure", "critical", asset_id, sid, placement,
                        last12[0][0], ts, 60, peak,
                        "Cabinet %s cooling failure: >9.5 C for 60 min (peak %.2f C)" % (asset_id, peak)
                    )
            else:
                self.active_alerts.pop((sid, "cabinet_failure"), None)

        # 5a. refrigeration_failure (Vehicle mid-trip: >= 12 slots on_board and last 6 slots > 10.0 C)
        if is_vehicle and active_now and len(hist) >= 12:
            last12 = hist[-12:]
            last6 = hist[-6:]
            if all(bool(ob) for _, _, ob in last12) and all(t is not None and t > 10.0 for _, t, _ in last6):
                key = (sid, "refrigeration_failure")
                if key not in self.active_alerts:
                    self.active_alerts[key] = last6[0][0]
                    peak = max(t for _, t, _ in last6)
                    self.publish_alert(
                        "refrigeration_failure", "critical", asset_id, sid, placement,
                        last6[0][0], ts, 30, peak,
                        "Vehicle %s reefer failure in transit: >10.0 C for 30 min (peak %.2f C)" % (asset_id, peak)
                    )
            else:
                self.active_alerts.pop((sid, "refrigeration_failure"), None)

        # 5b. refrigeration_failure (Depot Cold Room: both TS-CR-1 and TS-CR-2 > 7.0 C for >= 30 min = 6 slots)
        if asset_id == "DEPOT_COLDROOM" and len(self.history["TS-CR-1"]) >= 6 and len(self.history["TS-CR-2"]) >= 6:
            cr1 = list(self.history["TS-CR-1"])[-6:]
            cr2 = list(self.history["TS-CR-2"])[-6:]
            if (all(t is not None and t > 7.0 for _, t, _ in cr1) and
                    all(t is not None and t > 7.0 for _, t, _ in cr2)):
                key = ("DEPOT_COLDROOM", "refrigeration_failure")
                if key not in self.active_alerts:
                    self.active_alerts[key] = cr1[0][0]
                    peak = max([t for _, t, _ in cr1] + [t for _, t, _ in cr2])
                    self.publish_alert(
                        "refrigeration_failure", "critical", "DEPOT_COLDROOM", "TS-CR-1+TS-CR-2", "cold_room",
                        cr1[0][0], ts, 30, peak,
                        "Depot cold room refrigeration failure: both sensors >7.0 C for 30 min (peak %.2f C)" % peak
                    )
            else:
                self.active_alerts.pop(("DEPOT_COLDROOM", "refrigeration_failure"), None)


def run_live(args):
    client = make_client("coldchain-incident-detector")
    monitor = LiveIncidentMonitor(client)

    def on_connect(c, userdata, *cb_args):
        c.subscribe("coldchain/+/+/telemetry", qos=1)
        print("Connected to MQTT broker at %s:%d. Listening on coldchain/+/+/telemetry ..." % (
            args.host, args.port))
        print("Publishing alerts to coldchain/alerts/<asset_id> (Ctrl+C to stop)\n")

    def on_message(c, userdata, msg):
        try:
            m = json.loads(msg.payload)
            monitor.process_message(m)
        except Exception as e:
            print("Error processing message on %s: %s" % (msg.topic, e))

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nIncident detector stopped. Total alerts published: %d" % monitor.alert_counter)
    finally:
        client.disconnect()


def run_batch(args):
    print("Reading cleaned sensor data from %s ..." % args.file)
    df = mf.read_clean(args.file)
    inc_df = detect_all_batch(df)

    os.makedirs(os.path.dirname(OUT_INCIDENTS_CSV), exist_ok=True)
    inc_df.to_csv(OUT_INCIDENTS_CSV, index=False)

    print("\n=== INCIDENT DETECTION SUMMARY ===")
    print("Total incidents detected: %d\n" % len(inc_df))
    if not inc_df.empty:
        print("Counts by incident type:")
        print(inc_df["type"].value_counts().to_string())
        print("\nCounts by asset_id:")
        print(inc_df["asset_id"].value_counts().to_string())
        print("\nFirst 25 detected incidents:")
        cols = ["asset_id", "sensor_id", "type", "start", "end", "duration_min", "peak_temp_c"]
        print(inc_df[cols].head(25).to_string(index=False))

    # Write markdown summary table for easy inclusion in the report
    with open(OUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Detected Cold-Chain Incidents (1 Apr – 30 Sep 2025)\n\n")
        f.write("Total incidents detected: **%d**\n\n" % len(inc_df))
        if not inc_df.empty:
            f.write("## Summary by Incident Type\n\n")
            f.write("| Incident Type | Count |\n|---|---|\n")
            for k, v in inc_df["type"].value_counts().items():
                f.write("| `%s` | %d |\n" % (k, v))
            f.write("\n## Full Incident List (`asset_id`, `start`, `end`, `type`)\n\n")
            f.write("| Asset | Sensor | Type | Start | End | Duration (min) | Peak (°C) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for r in inc_df.itertuples(index=False):
                f.write("| %s | %s | `%s` | %s | %s | %d | %.2f |\n" % (
                    r.asset_id, r.sensor_id, r.type, r.start, r.end, r.duration_min, r.peak_temp_c))
    print("\nSaved full CSV list to: %s" % OUT_INCIDENTS_CSV)
    print("Saved Markdown table to: %s" % OUT_SUMMARY_MD)

    if args.publish_batch and not inc_df.empty:
        print("\nPublishing %d batch alerts to MQTT (%s:%d) ..." % (len(inc_df), args.host, args.port))
        client = make_client("coldchain-batch-alerter")
        client.connect(args.host, args.port, keepalive=60)
        client.loop_start()
        for idx, r in enumerate(inc_df.itertuples(index=False), start=1):
            payload = make_alert_payload(
                idx, r.type, r.severity, r.asset_id, r.sensor_id, r.placement,
                r.start, r.end, r.duration_min, r.peak_temp_c, r.details
            )
            info = client.publish(ALERT_TOPIC_FMT % r.asset_id, json.dumps(payload, separators=(",", ":")), qos=1)
            if idx % 200 == 0:
                info.wait_for_publish()
        info.wait_for_publish()
        client.loop_stop()
        client.disconnect()
        print("Published all %d alerts." % len(inc_df))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=mf.CLEAN_FILE, help="cleaned dataset path")
    ap.add_argument("--host", default="localhost", help="MQTT broker host")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--live", action="store_true", help="subscribe to live MQTT telemetry and publish alerts")
    ap.add_argument("--batch", action="store_true", help="scan cleaned CSV and generate incident list")
    ap.add_argument("--publish-batch", action="store_true", help="publish batch-detected alerts to MQTT")
    args = ap.parse_args()

    if args.live:
        run_live(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()