"""
slot_marker.py - Click on a video frame to mark parking slots.

Run:
    python -m src.slot_marker
    python -m src.slot_marker --video data/other.mp4 --frame 50
    python -m src.slot_marker --region 0 1440 1280 720   # zoom into one part

Controls (click inside the window first):
    Left click : add a corner point. After 4 points the slot is completed.
    u          : undo last point (or last slot)
    c          : clear all slots
    s          : save to data/slots.json
    q / Esc    : quit
"""
import argparse

import cv2
import numpy as np

from config import SLOTS_PATH, VIDEO_PATH
from src.detector import read_frame
from src.slots import load_slots, save_slots

WINDOW = "Slot marker  |  click 4 corners per slot  |  u=undo  c=clear  s=save  q=quit"
MAX_W, MAX_H = 1280, 800      # shrink big frames so they fit on a laptop screen


def main():
    parser = argparse.ArgumentParser(description="Mark parking slots on a video frame.")
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--frame", type=int, default=0, help="Frame number to mark on")
    parser.add_argument("--out", default=str(SLOTS_PATH))
    parser.add_argument("--fresh", action="store_true", help="Ignore existing slots.json")
    # NEW (A): mark only one part of the frame, shown zoomed in
    parser.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Mark only this part of the frame (original pixels), zoomed in")
    args = parser.parse_args()

    frame = read_frame(args.video, args.frame)
    h, w = frame.shape[:2]

    # NEW (B): work out which part of the frame to show
    if args.region:
        ox, oy, rw, rh = args.region
        rw, rh = min(rw, w - ox), min(rh, h - oy)   # keep the region inside the frame
    else:
        ox, oy, rw, rh = 0, 0, w, h                 # whole frame (old behaviour)
    scale = min(1.0, MAX_W / rw, MAX_H / rh)        # display scale; coordinates stay original

    slots, current = [], []
    if not args.fresh:
        try:
            data = load_slots(args.out)
            if tuple(data["frame_size"]) == (w, h):
                slots = [s["points"] for s in data["slots"]]
                print(f"Loaded {len(slots)} existing slots from {args.out}")
        except (FileNotFoundError, ValueError):
            pass

    saved = True

    def on_mouse(event, x, y, flags, param):
        nonlocal saved
        if event == cv2.EVENT_LBUTTONDOWN:
            # NEW (C): undo the zoom, then add the region offset -> full-frame pixels
            current.append([int(x / scale) + ox, int(y / scale) + oy])
            if len(current) == 4:
                slots.append(current.copy())
                current.clear()
            saved = False

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print(f"Frame size {w}x{h}. Showing region x={ox} y={oy} w={rw} h={rh}. "
          "Click 4 corners for each slot, then press 's' to save.")
    while True:
        view = frame.copy()
        for i, pts in enumerate(slots):
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(view, [arr], True, (0, 200, 0), 2)
            cx, cy = arr.mean(axis=0).astype(int)
            cv2.putText(view, f"P{i + 1}", (int(cx) - 12, int(cy) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for p in current:
            cv2.circle(view, tuple(p), 4, (0, 0, 255), -1)
        if len(current) > 1:
            cv2.polylines(view, [np.array(current, dtype=np.int32)], False, (0, 0, 255), 1)

        # NEW (D): crop to the region first, then shrink, then draw the status text
        view = view[oy:oy + rh, ox:ox + rw]
        if scale < 1.0:
            view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.putText(view, f"Slots: {len(slots)}  {'(saved)' if saved else '(unsaved)'}",
                    (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u"):
            if current:
                current.pop()
            elif slots:
                slots.pop()
            saved = False
        elif key == ord("c"):
            slots.clear()
            current.clear()
            saved = False
        elif key == ord("s"):
            if not slots:
                print("Nothing to save yet.")
            else:
                save_slots(args.out, slots, (w, h))
                saved = True
                print(f"Saved {len(slots)} slots to {args.out}")
        elif key in (ord("q"), 27):
            if not saved:
                print("Warning: you quit without saving the latest changes.")
            break

        # Stop if the user closed the window with the red button.
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()