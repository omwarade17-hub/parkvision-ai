"""
analysis.py - Clean the stored history and find occupancy patterns.

Run:
    python -m src.analysis --source synthetic
    python -m src.analysis --source video
Charts are saved to the outputs/ folder.
"""
import argparse

import matplotlib
matplotlib.use("Agg")            # save charts to files without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from config import DB_PATH, OUTPUT_DIR
from src.database import fetch_records

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def clean_records(df):
    """Remove impossible rows. Returns (clean_df, report dict)."""
    report = {"rows_in": len(df)}
    df = df.copy()

    report["missing_values"] = int(df.isna().any(axis=1).sum())
    df = df.dropna(subset=["timestamp", "total_slots", "occupied"])

    bad = (df["total_slots"] <= 0) | (df["occupied"] < 0) | (df["occupied"] > df["total_slots"])
    report["invalid_counts"] = int(bad.sum())
    df = df[~bad]

    dup = df.duplicated(subset=["timestamp", "source"])
    report["duplicates"] = int(dup.sum())
    df = df[~dup]

    # Recalculate derived columns so they always agree with the counts.
    df["vacant"] = df["total_slots"] - df["occupied"]
    df["occupancy_pct"] = (100 * df["occupied"] / df["total_slots"]).round(1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def hourly_profile(df):
    """Average occupancy % for each hour of the day (0-23)."""
    return df.groupby(df["timestamp"].dt.hour)["occupancy_pct"].mean().rename_axis("hour")


def weekday_profile(df):
    """Average occupancy % for each day of the week."""
    s = df.groupby(df["timestamp"].dt.dayofweek)["occupancy_pct"].mean()
    s.index = [DAY_NAMES[i] for i in s.index]
    return s


def hour_weekday_table(df):
    """Rows = weekday, columns = hour, values = mean occupancy %."""
    t = df.pivot_table(index=df["timestamp"].dt.dayofweek,
                       columns=df["timestamp"].dt.hour,
                       values="occupancy_pct", aggfunc="mean")
    t.index = [DAY_NAMES[i] for i in t.index]
    return t


def save_charts(df, label, out_dir=OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f" ({label.upper()} DATA)"
    paths = []

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ts = df.set_index("timestamp")["occupancy_pct"].resample("h").mean()
    ax.plot(ts.index, ts.values, linewidth=0.8)
    ax.set(title="Occupancy over time" + tag, ylabel="Occupancy %", ylim=(0, 100))
    fig.tight_layout(); p = out_dir / "timeline.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    hp = hourly_profile(df)
    ax.bar(hp.index, hp.values)
    ax.set(title="Average occupancy by hour" + tag, xlabel="Hour of day",
           ylabel="Occupancy %", ylim=(0, 100), xticks=range(0, 24, 2))
    fig.tight_layout(); p = out_dir / "hourly_pattern.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    wp = weekday_profile(df)
    ax.bar(wp.index, wp.values)
    ax.set(title="Average occupancy by weekday" + tag, ylabel="Occupancy %", ylim=(0, 100))
    fig.tight_layout(); p = out_dir / "weekday_pattern.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    table = hour_weekday_table(df)
    im = ax.imshow(table.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set(title="Occupancy heatmap: weekday x hour" + tag, xlabel="Hour of day")
    fig.colorbar(im, ax=ax, label="Occupancy %")
    fig.tight_layout(); p = out_dir / "heatmap.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Analyse stored occupancy history.")
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "video", "live", "all"])
    args = parser.parse_args()

    df = fetch_records(DB_PATH, args.source)
    if df.empty:
        raise SystemExit(f"No '{args.source}' records in {DB_PATH}. "
                         "Run run_detection or generate_synthetic first.")

    df, report = clean_records(df)
    print(f"Source: {args.source}" + ("   <-- SYNTHETIC, not real data" if args.source == "synthetic" else ""))
    print(f"Rows read: {report['rows_in']}, kept: {report['rows_out']} "
          f"(missing: {report['missing_values']}, invalid: {report['invalid_counts']}, "
          f"duplicates: {report['duplicates']})")
    print(f"Time range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Mean occupancy: {df['occupancy_pct'].mean():.1f}%")

    hp = hourly_profile(df)
    print(f"Busiest hour: {hp.idxmax()}:00 ({hp.max():.1f}%),  quietest: {hp.idxmin()}:00 ({hp.min():.1f}%)")
    print("By weekday (%):", weekday_profile(df).round(1).to_dict())

    for p in save_charts(df, args.source):
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
