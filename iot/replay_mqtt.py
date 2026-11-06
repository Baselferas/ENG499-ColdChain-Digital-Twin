"""
Replay the cleaned sensor readings over MQTT, one message per reading.

Format: see MQTT_message_format.md (topic coldchain/<asset_id>/<sensor_id>/telemetry,
JSON payload, QoS 1, retain off). The 'timestamp' in a message is always the original
dataset time; only the sending speed changes.

Examples
  python replay_mqtt.py --dry-run --limit 5                   # print messages, send nothing
  python replay_mqtt.py --start 2025-06-05 --end 2025-06-06   # one day, default speed
  python replay_mqtt.py --speed 600                            # 1 dataset hour = 6 seconds
  python replay_mqtt.py --fast                                 # as fast as the broker accepts
  python replay_mqtt.py --sensors TS-U07 TS-V3 --fast          # only some sensors

Speed: --speed N means the replay runs N times faster than real time. A 5-minute slot
(25 messages) then takes 300/N seconds. Default N = 3000 (0.1 s per slot, ~250 msg/s,
the full 6 months take about 90 minutes; use --fast or a time window for quick tests).
"""
import argparse
import sys
import time

import pandas as pd
import paho.mqtt.client as mqtt

import message_format as mf


def make_client(client_id):
    try:                                    # paho-mqtt 2.x
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:                  # paho-mqtt 1.x
        c = mqtt.Client(client_id=client_id)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=mf.CLEAN_FILE, help="cleaned file (default: %(default)s)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--start", help="first day/time to send, e.g. 2025-06-01")
    ap.add_argument("--end", help="last day/time to send (exclusive), e.g. 2025-06-08")
    ap.add_argument("--sensors", nargs="*", help="only these sensor ids")
    ap.add_argument("--speed", type=float, default=3000.0, help="times faster than real time")
    ap.add_argument("--fast", action="store_true", help="no waiting between readings")
    ap.add_argument("--limit", type=int, help="stop after this many messages")
    ap.add_argument("--dry-run", action="store_true", help="print messages, do not connect")
    args = ap.parse_args()

    df = mf.read_clean(args.file)
    if args.sensors:
        df = df[df["sensor_id"].isin(args.sensors)]

    t = pd.to_datetime(df["timestamp"])
    if args.start:
        df = df[t >= pd.Timestamp(args.start)]
        t = pd.to_datetime(df["timestamp"])
    if args.end:
        df = df[t < pd.Timestamp(args.end)]
    if df.empty:
        sys.exit("No readings selected.")
    df = mf.prepare(df)
    if args.limit:
        df = df.head(args.limit)
    print("%d messages, %s to %s" % (len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1]))

    client = None
    if not args.dry_run:
        client = make_client("coldchain-replay")
        client.max_inflight_messages_set(1000)
        client.max_queued_messages_set(0)
        client.connect(args.host, args.port, keepalive=60)
        client.loop_start()

    sent, bad, last_t, info = 0, 0, None, None
    t0 = time.time()
    try:
        for r in df.itertuples(index=False):
            msg = mf.row_to_message(r)
            problems = mf.check_message(msg)
            if problems:
                bad += 1
                print("WARNING", msg["sensor_id"], msg["timestamp"], problems, file=sys.stderr)
            if not args.fast and not args.dry_run and last_t is not None and r.dtime != last_t:
                time.sleep((r.dtime - last_t).total_seconds() / args.speed)
            last_t = r.dtime
            topic = mf.topic_for(msg["asset_id"], msg["sensor_id"])
            payload = mf.to_json(msg)
            if args.dry_run:
                print(topic, payload)
            else:
                info = client.publish(topic, payload, qos=1, retain=False)
            sent += 1
            if client is not None and sent % 500 == 0:
                info.wait_for_publish()             # wait for the broker's acknowledgement
            if sent % 20000 == 0:
                rate = sent / max(time.time() - t0, 1e-9)
                print("  %d / %d sent (%.0f msg/s), dataset time %s" % (sent, len(df), rate, msg["timestamp"]))
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        if client is not None:
            if info is not None:
                info.wait_for_publish()
            client.loop_stop()
            client.disconnect()
    print("done: %d messages %s, %d with problems, %.1f s" % (
        sent, "printed" if args.dry_run else "published", bad, time.time() - t0))


if __name__ == "__main__":
    main()
