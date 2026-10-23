"""
config.py - All project settings in one place.

Change values here instead of hunting through the code.
Every other module imports its paths and thresholds from this file.
""" 
from pathlib import Path

# 
# ---------------------------------------------------------------------------
# Folders and files
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent      # .../parkvision_ai
DATA_DIR = BASE_DIR / "data"                    # videos, slot map, database
MODELS_DIR = BASE_DIR / "models"                # trained prediction model
OUTPUT_DIR = BASE_DIR / "outputs"               # charts, latest frame/status

VIDEO_PATH = DATA_DIR / "parking2.mp4"           # default input video
SLOTS_PATH = DATA_DIR / "slots.json"            # parking-slot coordinates
DB_PATH = DATA_DIR / "parking.db"               # SQLite database

PRED_MODEL_PATH = MODELS_DIR / "occupancy_model.joblib"
LATEST_FRAME_PATH = OUTPUT_DIR / "latest_frame.jpg"     # read by dashboard
LATEST_STATUS_PATH = OUTPUT_DIR / "latest_status.json"  # read by dashboard

# ---------------------------------------------------------------------------
# YOLO vehicle detection
# ---------------------------------------------------------------------------
# "yolov8n.pt" is downloaded automatically (~6 MB) the first time it is used.
YOLO_WEIGHTS = "yolov8n-obb.pt"

# COCO class IDs that count as vehicles (YOLOv8n is trained on COCO).
VEHICLE_CLASSES =  {9: "large vehicle", 10: "small vehicle"}

CONF_THRESHOLD = 0.15        # ignore detections less confident than this
DETECT_EVERY_N_FRAMES = 3    # run YOLO on every 3rd frame (faster on a laptop)
# Size YOLO resizes each frame to before detecting (must be a multiple of 32).
# Default 640 is too small for 4K drone video: cars shrink to ~10 px and are missed.
IMG_SIZE = 1024
# YOLO returns at most this many detections per frame (default 300).
# This lot has 300+ cars, so the default cut detections off.
MAX_DETECTIONS = 1000
# ---------------------------------------------------------------------------
# Occupancy logic
# ---------------------------------------------------------------------------
# A slot is OCCUPIED when a vehicle's footprint covers at least this fraction
# of the smaller of (slot area, footprint area).
OVERLAP_THRESHOLD = 0.15

# Which part of the vehicle's bounding box is treated as its "footprint".
#   1.0 -> whole box   (best for top-down / drone-style views)
#   0.5 -> bottom half (better for angled CCTV views, where the top of the
#          box covers the slot *behind* the car)
FOOTPRINT_FRACTION = 1.0

# A slot must give the same reading this many detection cycles in a row
# before its displayed state changes. Stops flickering.
STABLE_FRAMES = 3

# ---------------------------------------------------------------------------
# Database logging
# ---------------------------------------------------------------------------
# Save one occupancy record every N seconds (not every frame -> no duplicates).
LOG_INTERVAL_SECONDS = 10
# How often the live frame + status are written for the dashboard (seconds).
# Separate from database logging, so the dashboard updates fast
# while the database still gets only one row every LOG_INTERVAL_SECONDS.
LIVE_UPDATE_SECONDS = 1


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
FORECAST_HOURS = 24          # maximum prediction horizon
MIN_HISTORY_DAYS = 14        # refuse to train with less history than this
