"""
database.py - SQLite storage for occupancy observations.

One table, one row per observation:

    occupancy_log
    ------------------------------------------------------------
    id            INTEGER  auto-numbered primary key
    timestamp     TEXT     "YYYY-MM-DD HH:MM:SS" (local time)
    total_slots   INTEGER
    occupied      INTEGER
    vacant        INTEGER
    occupancy_pct REAL     0-100
    source        TEXT     'live' | 'video' | 'synthetic'

The `source` column keeps real observations and made-up test data apart.
"""
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import DB_PATH

VALID_SOURCES = ("live", "video", "synthetic")

SCHEMA = """
CREATE TABLE IF NOT EXISTS occupancy_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    total_slots   INTEGER NOT NULL,
    occupied      INTEGER NOT NULL,
    vacant        INTEGER NOT NULL,
    occupancy_pct REAL    NOT NULL,
    source        TEXT    NOT NULL CHECK (source IN ('live', 'video', 'synthetic'))
);
CREATE INDEX IF NOT EXISTS idx_log_time ON occupancy_log (source, timestamp);
"""


def connect(db_path=DB_PATH):
    """Open (and create if needed) the database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def insert_record(conn, stats, source, timestamp=None):
    """Insert one observation. `stats` comes from occupancy.compute_stats()."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}")
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO occupancy_log "
        "(timestamp, total_slots, occupied, vacant, occupancy_pct, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",     # ? placeholders prevent SQL injection
        (ts, stats["total"], stats["occupied"], stats["vacant"],
         stats["occupancy_pct"], source),
    )
    conn.commit()


def insert_many(conn, rows):
    """Fast bulk insert. rows = list of tuples in column order (no id)."""
    conn.executemany(
        "INSERT INTO occupancy_log "
        "(timestamp, total_slots, occupied, vacant, occupancy_pct, source) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def fetch_records(db_path=DB_PATH, source=None):
    """Read observations into a pandas DataFrame, oldest first.

    source=None returns everything; otherwise 'live', 'video' or 'synthetic'.
    """
    conn = connect(db_path)
    try:
        query = "SELECT * FROM occupancy_log"
        params = ()
        if source and source != "all":
            query += " WHERE source = ?"
            params = (source,)
        query += " ORDER BY timestamp"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def delete_source(conn, source):
    """Delete all rows of one source (used to regenerate synthetic data)."""
    cur = conn.execute("DELETE FROM occupancy_log WHERE source = ?", (source,))
    conn.commit()
    return cur.rowcount


def count_by_source(db_path=DB_PATH):
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT source, COUNT(*), MIN(timestamp), MAX(timestamp) "
            "FROM occupancy_log GROUP BY source"
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: {"rows": r[1], "first": r[2], "last": r[3]} for r in rows}


if __name__ == "__main__":
    # Quick check:  python -m src.database
    summary = count_by_source()
    if not summary:
        print(f"Database {DB_PATH} is empty.")
    for src, info in summary.items():
        print(f"{src:<10} {info['rows']:>7} rows   {info['first']}  ->  {info['last']}")
