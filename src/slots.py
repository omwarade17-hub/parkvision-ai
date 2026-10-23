"""
slots.py - Load, save, validate and rescale parking-slot coordinates.

File format (data/slots.json):
{
  "frame_size": [1280, 720],                 # width, height of the video
  "slots": [
    {"id": 1, "points": [[x, y], [x, y], [x, y], [x, y]]},
    ...
  ]
}
"""
import json
from pathlib import Path


def save_slots(path, slots, frame_size):
    """Save a list of slots to a JSON file.

    slots      : list of point-lists, e.g. [[[10, 20], [50, 20], ...], ...]
    frame_size : (width, height) of the frame the points were clicked on
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "slots": [
            {"id": i + 1, "points": [[int(x), int(y)] for x, y in pts]}
            for i, pts in enumerate(slots)
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    return data


def load_slots(path):
    """Load and validate slots.json. Returns the dict described above."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Slot file not found: {path}\n"
            "Mark your parking slots first:  python -m src.slot_marker"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e

    validate_slots(data)
    return data


def validate_slots(data):
    """Raise ValueError if the slot data is unusable."""
    if "frame_size" not in data or "slots" not in data:
        raise ValueError("Slot file must contain 'frame_size' and 'slots'.")

    w, h = data["frame_size"]
    if w <= 0 or h <= 0:
        raise ValueError("frame_size must be positive.")
    if not data["slots"]:
        raise ValueError("Slot file contains no slots. Run the slot marker again.")

    seen_ids = set()
    for slot in data["slots"]:
        sid = slot.get("id")
        pts = slot.get("points", [])
        if sid in seen_ids:
            raise ValueError(f"Duplicate slot id: {sid}")
        seen_ids.add(sid)
        if len(pts) < 3:
            raise ValueError(f"Slot {sid} needs at least 3 points, has {len(pts)}.")
        for x, y in pts:
            # Allow a few pixels of slack for clicks right on the edge.
            if not (-5 <= x <= w + 5 and -5 <= y <= h + 5):
                raise ValueError(
                    f"Slot {sid} point ({x}, {y}) is outside the {w}x{h} frame."
                )


def scale_slots(data, new_size):
    """Return a copy of the slots rescaled to a different video resolution.

    Useful when slots were marked on a 1920x1080 frame but the video
    being processed is 1280x720 (same camera, different resolution).
    """
    old_w, old_h = data["frame_size"]
    new_w, new_h = new_size
    sx, sy = new_w / old_w, new_h / old_h
    return {
        "frame_size": [int(new_w), int(new_h)],
        "slots": [
            {"id": s["id"],
             "points": [[int(round(x * sx)), int(round(y * sy))] for x, y in s["points"]]}
            for s in data["slots"]
        ],
    }
