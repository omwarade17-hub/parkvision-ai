"""
dashboard.py - Streamlit dashboard for ParkVision AI.

Run (from the parkvision_ai folder, with the environment active):
    streamlit run dashboard.py
Stop with Ctrl + C in the terminal.

The dashboard only READS results:
  * current status  <- outputs/latest_status.json + latest_frame.jpg
                       (written by  python -m src.run_detection)
  * history         <- data/parking.db
  * prediction      <- models/occupancy_model.joblib  (python -m src.predict)
"""
import json

import joblib
import pandas as pd
import streamlit as st
import altair as alt

from config import (DB_PATH, FORECAST_HOURS, LATEST_FRAME_PATH,
                    LATEST_STATUS_PATH, PRED_MODEL_PATH)
from src.analysis import clean_records, hourly_profile, weekday_profile
from src.database import count_by_source, fetch_records
from src.predict import forecast, prepare_hourly

SOURCE_LABELS = {
    "video": "Recorded video (real detections, timestamped when processed)",
    "live": "Live camera (real detections)",
    "synthetic": "SYNTHETIC test data (not real observations)",
}

st.set_page_config(page_title="ParkVision AI", page_icon="🅿️", layout="wide")
st.title("🅿️ ParkVision AI — Smart Parking Dashboard")
st.caption("YOLOv8n vehicle detection • slot occupancy • SQLite history • ML forecast")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
available = count_by_source(DB_PATH)
with st.sidebar:
    st.header("Settings")
    options = [s for s in ("video", "live", "synthetic") if s in available] or ["synthetic"]
    source = st.selectbox("History / prediction data source", options,
                          format_func=lambda s: s.upper())
    st.caption(SOURCE_LABELS[source])
    if st.button("🔄 Refresh"):
        st.rerun()
    st.divider()
    st.subheader("Database")
    if available:
        for src, info in available.items():
            st.write(f"**{src}**: {info['rows']} rows")
    else:
        st.write("Empty")

# ---------------------------------------------------------------------------
# 1. Current status
# ---------------------------------------------------------------------------
st.header("1. Current occupancy")


@st.fragment(run_every="2s")
def show_current_status():
    """Auto-refreshes every 2 s: re-reads the latest status + frame written by run_detection."""
    if not LATEST_STATUS_PATH.exists():
        st.info("No live status yet. Start detection in another terminal:\n\n"
                "`python -m src.run_detection`")
        return
    try:
        status = json.loads(LATEST_STATUS_PATH.read_text())
    except json.JSONDecodeError:
        return                      # file was being replaced; try again in 2 s
    if not status:
        return

    kind = "LIVE camera" if status["source"] == "live" else "recorded VIDEO"
    st.caption(f"Last update {status['timestamp']} from {kind}: `{status['source_name']}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spaces", status["total"])
    c2.metric("Occupied", status["occupied"])
    c3.metric("Vacant", status["vacant"])
    c4.metric("Occupancy", f"{status['occupancy_pct']}%")
    st.progress(min(1.0, status["occupancy_pct"] / 100))

    left, right = st.columns([3, 1])
    if LATEST_FRAME_PATH.exists():
        # Pass bytes, not the path, so the newest image is always shown.
        left.image(LATEST_FRAME_PATH.read_bytes(), caption="Latest annotated frame",
                   width="stretch")
    slot_df = pd.DataFrame(
        [{"Slot": f"P{k}", "Status": v} for k, v in status["slots"].items()])
    right.dataframe(slot_df, hide_index=True, width="stretch")


show_current_status()
# ---------------------------------------------------------------------------
# 2. History
# ---------------------------------------------------------------------------
st.header("2. Historical occupancy")
df = fetch_records(DB_PATH, source)
if df.empty:
    st.info(f"No '{source}' records yet.")
else:
    df, report = clean_records(df)
    if source == "synthetic":
        st.warning("⚠️ This history is SYNTHETIC test data, generated to exercise the "
                   "pipeline. It does not describe a real car park.")
    st.caption(f"{report['rows_out']} records from {df['timestamp'].min()} to "
               f"{df['timestamp'].max()} ({report['rows_in'] - report['rows_out']} "
               "invalid/duplicate rows removed)")

    timeline = df.set_index("timestamp")["occupancy_pct"]
    if (timeline.index.max() - timeline.index.min()) > pd.Timedelta(days=2):
        timeline = timeline.resample("h").mean()
    st.subheader("Occupancy % over time")
    st.line_chart(timeline, height=260)

    a, b = st.columns(2)
    a.subheader("Average by hour of day")
    a.bar_chart(hourly_profile(df), height=240)
    b.subheader("Average by weekday")
    wp = weekday_profile(df).reset_index()
    wp.columns = ["Day", "Occupancy %"]
    chart = alt.Chart(wp).mark_bar().encode(
        x=alt.X("Day:N", sort=list(wp["Day"]), title=None),   # keep Mon..Sun order
        y=alt.Y("Occupancy %:Q", scale=alt.Scale(domain=[0, 100])),
        tooltip=["Day", alt.Tooltip("Occupancy %:Q", format=".1f")],
    ).properties(height=240)
    b.altair_chart(chart)
    b.caption("  ·  ".join(f"{d}: {v:.0f}%" for d, v in zip(wp["Day"], wp["Occupancy %"])))

# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------
st.header(f"3. Predicted occupancy (next {FORECAST_HOURS} hours)")
if not PRED_MODEL_PATH.exists():
    st.info(f"No trained model yet. Run:  `python -m src.predict --source {source}`")
elif df.empty:
    st.info("Need history to make a forecast.")
else:
    bundle = joblib.load(PRED_MODEL_PATH)
    if bundle.get("source") != source:
        st.warning(f"The saved model was trained on **{bundle.get('source')}** data, but "
                   f"you selected **{source}**. Retrain with "
                   f"`python -m src.predict --source {source}`.")
    else:
        series, total_slots = prepare_hourly(df)
        fc = forecast(bundle, series, total_slots, FORECAST_HOURS)
        if source == "synthetic":
            st.warning("⚠️ Model trained on SYNTHETIC data — this forecast demonstrates the "
                       "method only and is not a real-world prediction.")
        st.caption(f"Model: {bundle['model_name']} • trained {bundle['created_at']} • "
                   f"forecast starts after last record {series.index.max()}")

        nxt = fc.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Next hour occupancy", f"{nxt['predicted_occupancy_pct']}%")
        c2.metric("Next hour available", int(nxt["predicted_available"]))
        busiest = fc["predicted_occupancy_pct"].idxmax()
        c3.metric("Busiest hour ahead", busiest.strftime("%a %H:00"))

        st.line_chart(fc[["predicted_occupancy_pct"]], height=260)
        st.dataframe(
            fc.reset_index().assign(timestamp=lambda d: d["timestamp"].dt.strftime("%a %d %b %H:00"))
              .rename(columns={"timestamp": "Hour",
                               "predicted_occupancy_pct": "Occupancy %",
                               "predicted_occupied": "Occupied",
                               "predicted_available": "Available"}),
            hide_index=True, width="stretch")

        with st.expander("Model evaluation (test set = most recent 20% of history)"):
            st.dataframe(pd.DataFrame(bundle["metrics"]).T, width="stretch")
            st.caption("MAE/RMSE are in percentage points of occupancy. Baselines are simple "
                       "rules the ML model must beat to be worth using.")
            if not bundle["beats_baseline"]:
                st.error("The ML model did NOT beat the best baseline on this data.")
