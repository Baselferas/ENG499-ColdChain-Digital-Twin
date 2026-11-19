"""
Load the cleaned readings into the PostgreSQL / TimescaleDB database.

  python load_to_db.py                 # create the tables if needed and load the file
  python load_to_db.py --reset         # empty the readings table first
  python load_to_db.py --file other.csv.gz

Rows that already exist (same sensor_id and timestamp) are skipped, so the script can be
run again safely. Connection settings: --host/--port/--db/--user/--password or the
environment variables PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD (defaults: the values
in docker-compose.yml).
"""
import argparse
import io
import os
import sys
import time

import pandas as pd
import psycopg2

import message_format as mf

SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "schema.sql")
COLUMNS = ["ts", "sensor_id", "asset_id", "placement", "temperature_c", "raw_temperature_c",
           "humidity_pct", "battery_pct", "quality", "on_board"]


def connect(a):
    return psycopg2.connect(host=a.host, port=a.port, dbname=a.db, user=a.user, password=a.password)


def chunk_to_csv(df):
    """Convert a chunk of the clean file to CSV text for COPY (empty text = NULL)."""
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S") + mf.TZ_SUFFIX,
        "sensor_id": df["sensor_id"], "asset_id": df["asset_id"], "placement": df["placement"],
        "temperature_c": df["temperature_c"].round(2),
        "raw_temperature_c": df["raw_temperature_c"].round(2),
        "humidity_pct": df["humidity_pct"].round(2),
        "battery_pct": df["battery_pct"].round().astype("Int64"),
        "quality": df["quality"],
        "on_board": df["on_board"].map(mf._bool_or_none).map({True: "t", False: "f"}),
    })
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=mf.CLEAN_FILE)
    ap.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    ap.add_argument("--port", default=os.getenv("PGPORT", "5432"))
    ap.add_argument("--db", default=os.getenv("PGDATABASE", "coldchain"))
    ap.add_argument("--user", default=os.getenv("PGUSER", "coldchain"))
    ap.add_argument("--password", default=os.getenv("PGPASSWORD", "coldchain"))
    ap.add_argument("--reset", action="store_true", help="delete all readings before loading")
    ap.add_argument("--chunk", type=int, default=200_000)
    a = ap.parse_args()

    if not os.path.exists(a.file):
        sys.exit("Cleaned file not found: %s\nRun sensor_cleaning.py first." % a.file)
    conn = connect(a)
    cur = conn.cursor()
    with open(SCHEMA, encoding="utf-8") as f:
        cur.execute(f.read())
    if a.reset:
        cur.execute("TRUNCATE sensor_readings")
        print("readings table emptied")
    conn.commit()

    t0, total, inserted, sensors = time.time(), 0, 0, set()
    for chunk in mf.read_clean(a.file, chunksize=a.chunk):
        total += len(chunk)
        sensors |= set(map(tuple, chunk[["sensor_id", "asset_id", "placement"]].drop_duplicates().values))
        cur.execute("CREATE TEMP TABLE tmp_readings (LIKE sensor_readings INCLUDING DEFAULTS) ON COMMIT DROP")
        cur.copy_expert("COPY tmp_readings (%s) FROM STDIN WITH (FORMAT csv, NULL '')" % ", ".join(COLUMNS),
                        chunk_to_csv(chunk))
        cur.execute("INSERT INTO sensor_readings (%s) SELECT %s FROM tmp_readings "
                    "ON CONFLICT (sensor_id, ts) DO NOTHING" % (", ".join(COLUMNS), ", ".join(COLUMNS)))
        inserted += cur.rowcount
        conn.commit()
        print("  %d rows read, %d new rows inserted" % (total, inserted))
    cur.executemany("INSERT INTO sensors (sensor_id, asset_id, placement) VALUES (%s, %s, %s) "
                    "ON CONFLICT (sensor_id) DO UPDATE SET asset_id = EXCLUDED.asset_id, "
                    "placement = EXCLUDED.placement", sorted(sensors))
    conn.commit()

    print("\nchecks")
    cur.execute("SELECT count(*), count(DISTINCT sensor_id), min(ts), max(ts) FROM sensor_readings")
    n, ns, t_min, t_max = cur.fetchone()
    print("  rows in database: %d, sensors: %d, from %s to %s" % (n, ns, t_min, t_max))
    cur.execute("SELECT quality, count(*) FROM sensor_readings GROUP BY quality ORDER BY 2 DESC")
    for q, c in cur.fetchall():
        print("    %-14s %d" % (q, c))
    cur.execute("SELECT count(*) FROM sensor_readings WHERE quality <> 'gap' AND temperature_c IS NULL")
    print("  non-gap rows without temperature (must be 0):", cur.fetchone()[0])
    print("done in %.1f s" % (time.time() - t0))
    conn.close()


if __name__ == "__main__":
    main()
