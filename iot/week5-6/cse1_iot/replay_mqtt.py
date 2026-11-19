"""
Replay the cleaned sensor readings over MQTT as a continuous stream with adjustable speed.

Format:
  Topic:   coldchain/<asset_id>/<sensor_id>/telemetry
  Payload: JSON per reading, QoS 1, retain off
  Note:    'timestamp' in every message is always the original dataset time (+03:00).

Examples:
  # Continuous stream at 1 simulated day per real minute (speed = 1440x):
  python replay_mqtt.py --day-per-min 1 --loop

  # Continuous stream for a specific demo window (e.g. one day repeating every minute):
  python replay_mqtt.py --start 2025-07-15 --end 2025-07-16 --day-per-min 1 --loop

  # Single pass of one day at 600x speed (1 simulated hour = 6 seconds):
  python replay_mqtt.py --start 2025-06-05 --end 2025-06-06 --speed 600

  # Print first 5 messages without connecting to broker:
  python replay_mqtt.py --dry-run --limit 5
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
    ap.add_argument("--host", default="localhost", help="MQTT broker host (default: localhost)")
    ap.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    ap.add_argument("--start", help="first day/time to send, e.g. 2025-06-01")
    ap.add_argument("--end", help="last day/time to send (exclusive), e.g. 2025-06-08")
    ap.add_argument("--sensors", nargs="*", help="only these sensor ids")
    ap.add_argument("--speed", type=float, default=1440.0,
                    help="times faster than real time (default: 1440 = 1 sim day/min)")
    ap.add_argument("--day-per-min", type=float, default=None,
                    help="shortcut: set speed so N simulated days play per real minute (e.g. 1)")
    ap.add_argument("--loop", action="store_true",
                    help="run continuously in a loop until Ctrl+C")
    ap.add_argument("--fast", action="store_true", help="no waiting between readings")
    ap.add_argument("--limit", type=int, help="stop after this many messages")
    ap.add_argument("--dry-run", action="store_true", help="print messages, do not connect")
    args = ap.parse_args()

    speed = args.day_per_min * 1440.0 if args.day_per_min is not None else args.speed
    if speed <= 0:
        sys.exit("Speed must be > 0.")

    print("Loading %s ..." % args.file)
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

    sim_days = (df["dtime"].iloc[-1] - df["dtime"].iloc[0]).total_seconds() / 86400.0
    print("Loaded %d messages (%s to %s, %.2f simulated days)" % (
        len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1], sim_days))
    if args.fast:
        print("Mode: FAST (no delay) | Continuous loop: %s" % args.loop)
    else:
        print("Mode: %.1fx real-time (%.2f sim days/min, %.3f s per 5-min slot) | Continuous loop: %s" % (
            speed, speed / 1440.0, 300.0 / speed, args.loop))

    client = None
    if not args.dry_run:
        client = make_client("coldchain-replay")
        client.max_inflight_messages_set(1000)
        client.max_queued_messages_set(0)
        client.connect(args.host, args.port, keepalive=60)
        client.loop_start()

    sent, bad, cycle, info = 0, 0, 0, None
    t0 = time.time()
    try:
        while True:
            cycle += 1
            if args.loop:
                print("--- Starting stream cycle #%d ---" % cycle)
            last_t = None
            for r in df.itertuples(index=False):
                msg = mf.row_to_message(r)
                problems = mf.check_message(msg)
                if problems:
                    bad += 1
                    print("WARNING", msg["sensor_id"], msg["timestamp"], problems, file=sys.stderr)
                if not args.fast and not args.dry_run and last_t is not None and r.dtime != last_t:
                    dt_sec = (r.dtime - last_t).total_seconds()
                    if dt_sec > 0:
                        time.sleep(dt_sec / speed)
                last_t = r.dtime
                topic = mf.topic_for(msg["asset_id"], msg["sensor_id"])
                payload = mf.to_json(msg)
                if args.dry_run:
                    print(topic, payload)
                else:
                    info = client.publish(topic, payload, qos=1, retain=False)
                sent += 1
                if client is not None and sent % 500 == 0:
                    info.wait_for_publish()
                if sent % 7200 == 0:  # every simulated day (25 sensors * 288 slots = 7200 msgs)
                    rate = sent / max(time.time() - t0, 1e-9)
                    print("  [stream] %d msgs sent (%.0f msg/s) | sim_time: %s" % (
                        sent, rate, msg["timestamp"]))
            if not args.loop or args.dry_run:
                break
    except KeyboardInterrupt:
        print("\nStream stopped by user (Ctrl+C).")
    finally:
        if client is not None:
            if info is not None:
                info.wait_for_publish()
            client.loop_stop()
            client.disconnect()
    print("done: %d messages %s across %d cycle(s), %d with problems, %.1f s" % (
        sent, "printed" if args.dry_run else "published", cycle, bad, time.time() - t0))


if __name__ == "__main__":
    main()