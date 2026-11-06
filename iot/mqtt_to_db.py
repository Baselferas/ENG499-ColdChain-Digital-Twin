"""
MQTT subscriber that stores every telemetry message in the database.

Run it BEFORE replay_mqtt.py if you want the database to be filled from the live stream:
  python mqtt_to_db.py

It subscribes to coldchain/+/+/telemetry with QoS 1 and inserts the messages in small
batches. A message with the same sensor_id and timestamp that is already stored is ignored,
so it is safe to run this together with load_to_db.py. Stop with Ctrl+C.
"""
import argparse
import json
import os
import threading
import time

import paho.mqtt.client as mqtt
import psycopg2
from psycopg2.extras import execute_values

import message_format as mf

TOPIC = "coldchain/+/+/telemetry"
SQL = ("INSERT INTO sensor_readings (ts, sensor_id, asset_id, placement, temperature_c, "
       "raw_temperature_c, humidity_pct, battery_pct, quality, on_board) VALUES %s "
       "ON CONFLICT (sensor_id, ts) DO NOTHING")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    ap.add_argument("--port", default=os.getenv("PGPORT", "5432"))
    ap.add_argument("--db", default=os.getenv("PGDATABASE", "coldchain"))
    ap.add_argument("--user", default=os.getenv("PGUSER", "coldchain"))
    ap.add_argument("--password", default=os.getenv("PGPASSWORD", "coldchain"))
    a = ap.parse_args()

    conn = psycopg2.connect(host=a.host, port=a.port, dbname=a.db, user=a.user, password=a.password)
    buffer, lock, stats = [], threading.Lock(), {"received": 0, "rejected": 0, "new": 0}

    def on_message(client, userdata, msg):
        try:
            m = json.loads(msg.payload)
            if mf.check_message(m):
                raise ValueError(mf.check_message(m))
            row = (m["timestamp"], m["sensor_id"], m["asset_id"], m["placement"], m["temperature_c"],
                   m["raw_temperature_c"], m["humidity_pct"], m["battery_pct"], m["quality"], m["on_board"])
        except Exception as e:
            stats["rejected"] += 1
            print("rejected message on %s: %s" % (msg.topic, e))
            return
        with lock:
            buffer.append(row)
            stats["received"] += 1

    def on_connect(client, userdata, *args):
        client.subscribe(TOPIC, qos=1)
        print("connected to MQTT, waiting for messages on", TOPIC)

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="coldchain-db-writer",
                             clean_session=False)
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id="coldchain-db-writer", clean_session=False)
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(a.mqtt_host, a.mqtt_port, keepalive=60)
    client.loop_start()

    def flush():
        with lock:
            rows = buffer[:]
            buffer.clear()
        if rows:
            with conn.cursor() as cur:
                execute_values(cur, SQL, rows, page_size=2000)
                stats["new"] += cur.rowcount if cur.rowcount > 0 else 0
            conn.commit()

    last_print = time.time()
    try:
        while True:
            time.sleep(1.0)
            flush()
            if time.time() - last_print > 10:
                print("received %(received)d, rejected %(rejected)d, stored (last batch) %(new)d" % stats)
                last_print = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        flush()
        conn.close()
        print("stopped. received %(received)d, rejected %(rejected)d" % stats)


if __name__ == "__main__":
    main()
