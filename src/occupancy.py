"""
occupancy.py - Decide which parking slots are OCCUPIED or VACANT.

Idea in one sentence:
    For every detected vehicle, find the ONE slot its footprint overlaps the
    most; if that overlap is big enough, mark the slot as occupied.

Assigning each vehicle to only its best slot stops one car from being
counted in two neighbouring slots.
"""
import cv2
import numpy as np

from config import FOOTPRINT_FRACTION, OVERLAP_THRESHOLD, STABLE_FRAMES

GREEN = (0, 200, 0)      # OpenCV uses BGR order, not RGB
RED = (0, 0, 230)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def slot_polygon(points):
    """Turn clicked points into a convex polygon (float32 array, shape N x 2).

    convexHull fixes the order if the user clicked corners in a zig-zag.
    """
    pts = np.array(points, dtype=np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)


def footprint_polygon(box, fraction=FOOTPRINT_FRACTION):
    """The part of a bounding box that touches the ground.

    box      : (x1, y1, x2, y2)
    fraction : 1.0 = whole box, 0.5 = bottom half of the box
    """
    x1, y1, x2, y2 = box
    top = y2 - (y2 - y1) * fraction
    return np.array([[x1, top], [x2, top], [x2, y2], [x1, y2]], dtype=np.float32)


def polygon_area(poly):
    return float(cv2.contourArea(poly))


def intersection_area(poly_a, poly_b):
    """Area shared by two convex polygons (0 if they don't touch)."""
    area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    return max(float(area), 0.0)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def find_occupied_slots(slots, detections,
                        threshold=OVERLAP_THRESHOLD,
                        footprint_fraction=FOOTPRINT_FRACTION):
    """Return (set of occupied slot ids, list of matches).

    slots      : list of {"id": int, "points": [[x, y], ...]}
    detections : list of {"box": (x1, y1, x2, y2), ...}
    matches    : list of (detection, slot_id, overlap_score) for debugging
    """
    polys = {s["id"]: slot_polygon(s["points"]) for s in slots}
    areas = {sid: polygon_area(p) for sid, p in polys.items()}

    occupied, matches = set(), []
    for det in detections:
        fp = footprint_polygon(det["box"], footprint_fraction)
        fp_area = polygon_area(fp)
        if fp_area <= 0:
            continue

        best_id, best_score = None, 0.0
        for sid, poly in polys.items():
            if areas[sid] <= 0:
                continue
            inter = intersection_area(poly, fp)
            if inter <= 0:
                continue
            # Divide by the smaller area so that both a small motorbike in a
            # big slot and a big truck over a small slot score fairly.
            score = inter / min(areas[sid], fp_area)
            if score > best_score:
                best_id, best_score = sid, score

        if best_id is not None and best_score >= threshold:
            occupied.add(best_id)
            matches.append((det, best_id, round(best_score, 3)))

    return occupied, matches


def compute_stats(total, occupied_count):
    """Counts and percentages for the current frame."""
    occupied_count = max(0, min(occupied_count, total))
    vacant = total - occupied_count
    occ_pct = round(100.0 * occupied_count / total, 1) if total else 0.0
    return {
        "total": total,
        "occupied": occupied_count,
        "vacant": vacant,
        "occupancy_pct": occ_pct,
        "availability_pct": round(100.0 - occ_pct, 1) if total else 0.0,
    }


class SlotSmoother:
    """Only change a slot's state after it disagrees `required` times in a row.

    Example with required=3:
        raw readings : OCC OCC VAC OCC OCC   -> stays OCC (one glitch ignored)
        raw readings : OCC VAC VAC VAC       -> becomes VAC on the 3rd VAC
    """

    def __init__(self, slot_ids, required=STABLE_FRAMES):
        self.required = max(1, int(required))
        self.state = {sid: False for sid in slot_ids}
        self.counter = {sid: 0 for sid in slot_ids}
        self.initialized = False

    def update(self, raw_occupied):
        if not self.initialized:            # first reading: trust it directly
            for sid in self.state:
                self.state[sid] = sid in raw_occupied
            self.initialized = True
        else:
            for sid in self.state:
                raw = sid in raw_occupied
                if raw != self.state[sid]:
                    self.counter[sid] += 1
                    if self.counter[sid] >= self.required:
                        self.state[sid] = raw
                        self.counter[sid] = 0
                else:
                    self.counter[sid] = 0
        return {sid for sid, occ in self.state.items() if occ}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_overlay(frame, slots, occupied_ids, stats, detections=None):
    """Draw coloured slots, labels, vehicle boxes and a stats panel."""
    out = frame.copy()
    fill = frame.copy()

    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32)
        color = RED if slot["id"] in occupied_ids else GREEN
        cv2.fillPoly(fill, [pts], color)

    # Blend the filled polygons so the cars stay visible underneath.
    cv2.addWeighted(fill, 0.25, out, 0.75, 0, dst=out)

    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32)
        occ = slot["id"] in occupied_ids
        color = RED if occ else GREEN
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        label = f"P{slot['id']} {'OCC' if occ else 'VAC'}"
        cv2.putText(out, label, (int(cx) - 30, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 2, cv2.LINE_AA)
        cv2.putText(out, label, (int(cx) - 30, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for det in detections or []:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 1)

    lines = [
        f"Total:    {stats['total']}",
        f"Occupied: {stats['occupied']}",
        f"Vacant:   {stats['vacant']}",
        f"Occupancy: {stats['occupancy_pct']}%",
    ]
    cv2.rectangle(out, (10, 10), (230, 30 + 25 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (20, 35 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return out
