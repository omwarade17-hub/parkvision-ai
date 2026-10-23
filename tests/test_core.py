"""
test_core.py - Automatic checks for the core logic (no YOLO, no internet needed).

Run from the parkvision_ai folder:
    python -m tests.test_core

Each test_* function checks one behaviour. A FakeDetector finds coloured
rectangles in a generated video, so the whole pipeline (slots -> occupancy
-> database -> analysis -> prediction) can be tested without the real model.
"""
import json
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

from src import database
from src.analysis import clean_records
from src.generate_synthetic import generate_rows
from src.occupancy import SlotSmoother, compute_stats, find_occupied_slots
from src.predict import (InsufficientDataError, forecast, prepare_hourly,
                         train_and_evaluate)
from src.run_detection import process_stream
from src.slots import load_slots, save_slots, scale_slots

# Three side-by-side 100x200 slots.
SLOTS = [
    {"id": 1, "points": [[0, 0], [100, 0], [100, 200], [0, 200]]},
    {"id": 2, "points": [[100, 0], [200, 0], [200, 200], [100, 200]]},
    {"id": 3, "points": [[200, 0], [300, 0], [300, 200], [200, 200]]},
]


# ---------------------------------------------------------------------------
# Occupancy logic
# ---------------------------------------------------------------------------
def test_car_inside_one_slot():
    occ, _ = find_occupied_slots(SLOTS, [{"box": (10, 20, 90, 180)}])
    assert occ == {1}, occ


def test_car_across_two_slots_counts_once():
    # Mostly in slot 2, a little in slot 3 -> only slot 2.
    occ, _ = find_occupied_slots(SLOTS, [{"box": (110, 20, 230, 180)}])
    assert occ == {2}, occ


def test_vehicle_outside_slots_ignored():
    # A box far outside every slot must not count.
    occ2, _ = find_occupied_slots(SLOTS, [{"box": (400, 400, 500, 500)}])
    assert occ2 == set(), occ2


def test_stats():
    s = compute_stats(20, 5)
    assert s == {"total": 20, "occupied": 5, "vacant": 15,
                 "occupancy_pct": 25.0, "availability_pct": 75.0}, s
    assert compute_stats(0, 0)["occupancy_pct"] == 0.0
    assert compute_stats(10, 15)["occupied"] == 10      # clipped to capacity


def test_smoother_ignores_single_glitch():
    sm = SlotSmoother([1], required=3)
    assert sm.update({1}) == {1}
    assert sm.update(set()) == {1}       # one glitch
    assert sm.update({1}) == {1}
    for _ in range(2):
        assert sm.update(set()) == {1}
    assert sm.update(set()) == set()     # third vacant in a row -> changes


# ---------------------------------------------------------------------------
# Slot file
# ---------------------------------------------------------------------------
def test_slot_file_roundtrip_and_scaling():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "slots.json"
        save_slots(path, [s["points"] for s in SLOTS], (300, 200))
        data = load_slots(path)
        assert len(data["slots"]) == 3
        scaled = scale_slots(data, (600, 400))
        assert scaled["slots"][2]["points"][1] == [600, 0]


def test_bad_slot_file_rejected():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "slots.json"
        path.write_text(json.dumps({"frame_size": [100, 100],
                                    "slots": [{"id": 1, "points": [[0, 0], [500, 0], [0, 50]]}]}))
        try:
            load_slots(path)
        except ValueError:
            return
        raise AssertionError("out-of-frame slot was accepted")


# ---------------------------------------------------------------------------
# End-to-end with a generated video
# ---------------------------------------------------------------------------
class FakeDetector:
    """Detects pure-red rectangles as 'cars'."""

    def detect(self, frame):
        mask = cv2.inRange(frame, (0, 0, 200), (60, 60, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 500:
                dets.append({"box": (x, y, x + w, y + h), "class_id": 2,
                             "label": "car", "confidence": 1.0})
        return dets


def make_video(path, frames=30):
    w, h = 300, 200
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), 90, np.uint8)
        for x in (100, 200):
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), 2)
        cv2.rectangle(img, (15, 30), (85, 170), (0, 0, 255), -1)       # car in P1
        if i >= 15:
            cv2.rectangle(img, (215, 30), (285, 170), (0, 0, 255), -1)  # car arrives in P3
        vw.write(img)
    vw.release()


def test_pipeline_on_generated_video():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        video = d / "test.mp4"
        make_video(video)
        assert video.stat().st_size > 0, "video writer produced an empty file"
        slots = {"frame_size": [300, 200], "slots": SLOTS}
        db = d / "test.db"
        out = d / "out"

        res = process_stream(str(video), slots, FakeDetector(), show=False,
                             db_path=db, output_dir=out, log_interval=0, detect_every=1, live_interval=0)
        assert res["frames"] == 30, res
        assert res["occupied"] == [1, 3], res
        assert res["stats"]["occupancy_pct"] == 66.7, res

        df = database.fetch_records(db, "video")
        assert len(df) == 30 and df["occupied"].iloc[0] == 1 and df["occupied"].iloc[-1] == 2
        status = json.loads((out / "latest_status.json").read_text())
        assert status["slots"] == {"1": "OCCUPIED", "2": "VACANT", "3": "OCCUPIED"}
        assert (out / "latest_frame.jpg").exists()


# ---------------------------------------------------------------------------
# Database, analysis, prediction
# ---------------------------------------------------------------------------
def test_cleaning_removes_bad_rows():
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:00",
                                     "2026-01-01 11:00", "2026-01-01 12:00"]),
        "total_slots": [10, 10, 10, 10], "occupied": [5, 5, 12, 3],
        "vacant": [5, 5, -2, 7], "occupancy_pct": [50, 50, 120, 30],
        "source": ["video"] * 4})
    clean, rep = clean_records(df)
    assert len(clean) == 2 and rep["invalid_counts"] == 1 and rep["duplicates"] == 1


def test_prediction_refuses_short_history():
    with tempfile.TemporaryDirectory() as d:
        conn = database.connect(Path(d) / "t.db")
        database.insert_many(conn, generate_rows(20, days=3, interval_min=15))
        conn.close()
        series, _ = prepare_hourly(database.fetch_records(Path(d) / "t.db", "synthetic"))
        try:
            train_and_evaluate(series)
        except InsufficientDataError:
            return
        raise AssertionError("model trained on only 3 days")


def test_prediction_on_synthetic():
    with tempfile.TemporaryDirectory() as d:
        conn = database.connect(Path(d) / "t.db")
        database.insert_many(conn, generate_rows(20, days=56, interval_min=15))
        conn.close()
        series, total = prepare_hourly(database.fetch_records(Path(d) / "t.db", "synthetic"))
        bundle = train_and_evaluate(series)
        fc = forecast(bundle, series, total, 24)
        assert len(fc) == 24
        assert fc["predicted_occupancy_pct"].between(0, 100).all()
        assert ((fc["predicted_occupied"] + fc["predicted_available"]) == total).all()
        print(f"      metrics: {bundle['metrics']}")


# ---------------------------------------------------------------------------
def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
