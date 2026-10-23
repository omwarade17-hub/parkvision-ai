"""
generate_synthetic.py - Create clearly-labelled SYNTHETIC history for testing.

Why: a few minutes of video cannot teach a model about Monday mornings or
Sunday evenings. To build and test the analysis + ML pipeline we generate
made-up but realistic-looking data. Every row is stored with
source='synthetic' so it can never be confused with real observations.

Run:
    python -m src.generate_synthetic                 # 8 weeks, 15-min interval
    python -m src.generate_synthetic --days 30 --clear
"""
import argparse
from datetime import datetime, timedelta

import numpy as np

from config import DB_PATH, SLOTS_PATH
from src import database
from src.slots import load_slots


def expected_occupancy(hour, weekday):
    """Smooth 'typical' occupancy fraction (0-1) for an hour and weekday.

    Weekdays: morning peak ~10:30 and afternoon peak ~16:00 (office-style).
    Weekends: one broad midday peak (shopping-style), lower overall.
    """
    if weekday < 5:
        p = (0.08 + 0.78 * np.exp(-((hour - 10.5) / 2.8) ** 2)
             + 0.55 * np.exp(-((hour - 16.0) / 2.2) ** 2))
    else:
        p = 0.06 + 0.55 * np.exp(-((hour - 13.0) / 3.5) ** 2)
    return float(min(p, 0.97))


def generate_rows(total_slots, days, interval_min, end=None, seed=42):
    rng = np.random.default_rng(seed)
    end = (end or datetime.now()).replace(second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % interval_min)
    start = end - timedelta(days=days)

    rows, t = [], start
    day_factor, current_day = 1.0, None
    while t <= end:
        if t.date() != current_day:                       # busy / quiet days
            current_day = t.date()
            day_factor = rng.uniform(0.85, 1.12)
        hour = t.hour + t.minute / 60
        p = expected_occupancy(hour, t.weekday()) * day_factor
        p = float(np.clip(p + rng.normal(0, 0.05), 0, 1))
        occupied = int(round(p * total_slots))
        vacant = total_slots - occupied
        pct = round(100 * occupied / total_slots, 1)
        rows.append((t.strftime("%Y-%m-%d %H:%M:%S"), total_slots, occupied,
                     vacant, pct, "synthetic"))
        t += timedelta(minutes=interval_min)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC occupancy history.")
    parser.add_argument("--days", type=int, default=56)
    parser.add_argument("--interval", type=int, default=15, help="Minutes between rows")
    parser.add_argument("--total", type=int, default=None,
                        help="Total slots (default: number of slots in slots.json, else 20)")
    parser.add_argument("--clear", action="store_true", help="Delete old synthetic rows first")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.total
    if total is None:
        try:
            total = len(load_slots(SLOTS_PATH)["slots"])
        except (FileNotFoundError, ValueError):
            total = 20
    if total <= 0:
        raise SystemExit("--total must be positive")

    conn = database.connect(DB_PATH)
    try:
        if args.clear:
            removed = database.delete_source(conn, "synthetic")
            print(f"Removed {removed} old synthetic rows.")
        rows = generate_rows(total, args.days, args.interval, seed=args.seed)
        database.insert_many(conn, rows)
    finally:
        conn.close()

    print(f"Inserted {len(rows)} SYNTHETIC rows ({args.days} days, {total} slots, "
          f"every {args.interval} min). These are NOT real observations.")


if __name__ == "__main__":
    main()
