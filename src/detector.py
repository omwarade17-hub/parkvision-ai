"""
detector.py - Vehicle detection with a pretrained YOLOv8n model.

Run on its own to test detection on one image or video frame:
    python -m src.detector                       # first frame of data/parking.mp4
    python -m src.detector --image data/car.jpg  # any image
"""
import argparse

import cv2

from config import (CONF_THRESHOLD, OUTPUT_DIR, VEHICLE_CLASSES, VIDEO_PATH,
                    YOLO_WEIGHTS,IMG_SIZE,MAX_DETECTIONS)


class VehicleDetector:
    """Wraps YOLOv8n and returns only vehicle detections."""

    def __init__(self, weights=YOLO_WEIGHTS, conf=CONF_THRESHOLD,
                 classes=None, device=None, imgsz=IMG_SIZE):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Run:\n"
                "    python -m pip install ultralytics"
            ) from e

        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes or VEHICLE_CLASSES
        self.device = device            # None = auto, "cpu", or "mps" (Apple GPU)
        # First run downloads yolov8n.pt from the internet (~6 MB).
        self.model = YOLO(weights)

    def detect(self, frame):
        """Return a list of dicts: box (x1, y1, x2, y2), class_id, label, confidence."""
        kwargs = {"conf": self.conf, "classes": list(self.classes),
             "imgsz": self.imgsz, "verbose": False,"max_det": MAX_DETECTIONS}
        if self.device:
            kwargs["device"] = self.device
        result = self.model(frame, **kwargs)[0]
        boxes = result.obb if result.obb is not None else result.boxes

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append({
                "box": (x1, y1, x2, y2),
                "class_id": cls_id,
                "label": self.classes.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf[0]), 3),
            })
        return detections


def read_frame(video_path, index=0):
    """Read a single frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {index} from {video_path}")
    return frame


def draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(out, f"{d['label']} {d['confidence']:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8n vehicle detection.")
    parser.add_argument("--image", help="Image file (default: first frame of the video)")
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--device", default=None, help="cpu or mps")
    parser.add_argument("--no-show", action="store_true", help="Don't open a window")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
    else:
        frame = read_frame(args.video)

    detector = VehicleDetector(device=args.device)
    detections = detector.detect(frame)

    print(f"Found {len(detections)} vehicle(s):")
    for d in detections:
        print(f"  {d['label']:<10} conf={d['confidence']:.2f}  box={d['box']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "detection_test.jpg"
    annotated = draw_detections(frame, detections)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")

    if not args.no_show:
        cv2.imshow("Detection test (press any key)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
