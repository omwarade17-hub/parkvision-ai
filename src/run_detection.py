"""
run_detection.py - The main live/recorded occupancy pipeline.

    video frame -> YOLOv8n vehicles -> match to slots -> smooth -> draw
                -> every LOG_INTERVAL_SECONDS: save to SQLite + latest frame/status

Run:
    python -m src.run_detection                       # data/parking.mp4
    python -m src.run_detection --source 0            # webcam
    python -m src.run_detection --device mps --loop   # Apple GPU, repeat video
Keys in the window:  q = quit,  p = pause/resume
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import (DB_PATH, DETECT_EVERY_N_FRAMES, LOG_INTERVAL_SECONDS, LIVE_UPDATE_SECONDS,
                    OUTPUT_DIR, SLOTS_PATH, VIDEO_PATH)
from src import database
from src.occupancy import (SlotSmoother, compute_stats, draw_overlay,
                           find_occupied_slots)
from src.slots import load_slots, scale_slots

WINDOW = "ParkVision AI  |  q = quit  p = pause"


def parse_source(source):
    """Return (value for cv2.VideoCapture, source label for the database)."""
    s = str(source)
    if s.isdigit():
        return int(s), "live"                  # webcam index
    if s.lower().startswith(("rtsp://", "http://", "https://")):
        return s, "live"                       # CCTV / IP camera stream
    if not Path(s).exists():
        raise FileNotFoundError(
            f"Video not found: {s}\nPut your video at {VIDEO_PATH} or pass --source"
        )
    return s, "video"


def _atomic_write_status(status, frame, output_dir):
    """Write latest status + frame so the dashboard never reads half a file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_img = output_dir / "latest_frame.tmp.jpg"
    cv2.imwrite(str(tmp_img), frame)
    os.replace(tmp_img, output_dir / "latest_frame.jpg")
    tmp_json = output_dir / "latest_status.tmp.json"
    tmp_json.write_text(json.dumps(status, indent=2))
    os.replace(tmp_json, output_dir / "latest_status.json")


def process_stream(source, slots_data, detector, *, show=True, use_db=True,
                   db_path=DB_PATH, output_dir=OUTPUT_DIR, max_frames=None,
                   loop=False, log_interval=LOG_INTERVAL_SECONDS,live_interval=LIVE_UPDATE_SECONDS,
                   detect_every=DETECT_EVERY_N_FRAMES):
    """Run the full pipeline. `detector` is anything with a .detect(frame) method."""
    cap_source, source_label = parse_source(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {source}  ({w}x{h} @ {fps:.1f} fps, type={source_label})")

    if tuple(slots_data["frame_size"]) != (w, h):
        print(f"Note: slots were marked on {slots_data['frame_size']}, "
              f"video is {w}x{h}. Rescaling slot coordinates.")
        slots_data = scale_slots(slots_data, (w, h))
    slots = slots_data["slots"]
    smoother = SlotSmoother([s["id"] for s in slots])

    conn = database.connect(db_path) if use_db else None
    frame_idx, logged = 0, 0
    last_log = -float("inf")
    last_live = -float("inf")
    detections, occupied, stats = [], set(), compute_stats(len(slots), 0)
    paused = False

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if loop and source_label == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("End of video.")
                    break

                if frame_idx % detect_every == 0:
                    detections = detector.detect(frame)
                    raw_occupied, _ = find_occupied_slots(slots, detections)
                    occupied = smoother.update(raw_occupied)
                    stats = compute_stats(len(slots), len(occupied))

                annotated = draw_overlay(frame, slots, occupied, stats, detections)

                now = time.time()

                # Live view for the dashboard: every LIVE_UPDATE_SECONDS.
                if now - last_live >= live_interval:
                    last_live = now
                    status = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": source_label,
                        "source_name": str(source),
                        **stats,
                        "slots": {str(s["id"]): ("OCCUPIED" if s["id"] in occupied
                                                 else "VACANT") for s in slots},
                    }
                    _atomic_write_status(status, annotated, output_dir)

                # History for charts/prediction: every LOG_INTERVAL_SECONDS.
                if now - last_log >= log_interval:
                    last_log = now
                    if conn is not None:
                        database.insert_record(conn, stats, source_label)
                    logged += 1

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break

            if show:
                cv2.imshow(WINDOW, annotated)
                key = cv2.waitKey(1 if not paused else 50) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = not paused
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if conn is not None:
            conn.close()

    print(f"Processed {frame_idx} frames, saved {logged} records. Last: {stats}")
    return {"frames": frame_idx, "records": logged, "stats": stats,
            "occupied": sorted(occupied)}


def main():
    parser = argparse.ArgumentParser(description="Detect parking occupancy.")
    parser.add_argument("--source", default=str(VIDEO_PATH),
                        help="Video path, webcam index (0) or rtsp:// URL")
    parser.add_argument("--slots", default=str(SLOTS_PATH))
    parser.add_argument("--device", default=None, help="cpu or mps (Apple GPU)")
    parser.add_argument("--no-show", action="store_true", help="Run without a window")
    parser.add_argument("--no-db", action="store_true", help="Don't save to the database")
    parser.add_argument("--loop", action="store_true", help="Restart video at the end")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    slots_data = load_slots(args.slots)

    from src.detector import VehicleDetector   # imported here: loads YOLO only when needed
    detector = VehicleDetector(device=args.device)

    process_stream(args.source, slots_data, detector, show=not args.no_show,
                   use_db=not args.no_db, max_frames=args.max_frames, loop=args.loop)


if __name__ == "__main__":
    main()
