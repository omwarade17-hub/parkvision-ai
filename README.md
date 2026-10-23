# ParkVision AI
**AI-Based Smart Parking Space Detection and Occupancy Prediction System Using Computer Vision and Machine Learning**

The system watches a parking area from a fixed camera, detects vehicles with a YOLOv8n
model trained on aerial images (YOLOv8n-OBB, DOTA dataset), decides which marked parking
slots are OCCUPIED or VACANT, logs occupancy to SQLite, analyses historical patterns,
forecasts occupancy for the next 24 hours, and shows everything on a live Streamlit
dashboard.

---

## 1. Architecture

```
 Video / webcam / CCTV
        │  (OpenCV reads frames)
        ▼
 YOLOv8n-OBB vehicle detection ──► boxes for "small vehicle" / "large vehicle"
        │  (imgsz 1024–1920, max 1000 detections per frame)
        ▼
 Slot matching  ◄── data/slots.json  (marked once with slot_marker.py)
        │  each vehicle → the ONE slot it overlaps most
        ▼
 Smoothing (state must repeat 3 times) → OCCUPIED / VACANT per slot
        │
        ├──► Annotated video window (red = occupied, green = vacant)
        ├──► outputs/latest_status.json + latest_frame.jpg  (every 1 s, for the dashboard)
        └──► SQLite data/parking.db  (one row every 10 s)
                    │
                    ▼
        Historical analysis (Pandas + Matplotlib / Altair)
                    │
                    ▼
        ML forecast (Scikit-learn: baselines vs Linear Regression vs Random Forest)
                    │
                    ▼
        Streamlit dashboard (current-status section auto-refreshes every 2 s)
```

**Current detection** (Computer Vision) and **future prediction** (Machine Learning)
are separate parts. CV answers "what is happening now?"; ML answers "what will
probably happen later?" based on stored history.

---

## 2. Folder structure

| Path | Purpose |
|---|---|
| `config.py` | All settings: paths, model, thresholds, intervals |
| `src/detector.py` | YOLO wrapper; returns only vehicle boxes (supports normal and OBB models). Can test one image |
| `src/slot_marker.py` | Click-to-mark tool for parking slots → `data/slots.json`; `--region` zooms into part of a large frame |
| `src/slots.py` | Load / save / validate / rescale slot coordinates |
| `src/occupancy.py` | Overlap logic, smoothing, counts, drawing |
| `src/run_detection.py` | Main pipeline: video → detection → occupancy → live status + database |
| `src/database.py` | SQLite table, insert and read functions |
| `src/generate_synthetic.py` | Clearly-labelled synthetic history for testing ML |
| `src/analysis.py` | Data cleaning, hourly / weekday patterns, charts |
| `src/predict.py` | Feature preparation, time-ordered evaluation, forecast |
| `dashboard.py` | Streamlit dashboard |
| `tests/test_core.py` | 11 automatic tests (no YOLO or internet needed) |
| `data/` | Your video, `slots.json`, `parking.db` (video and database are not in git) |
| `models/` | Saved prediction model (recreated by `src.predict`) |
| `outputs/` | Charts, latest frame and status |

---

## 3. Setup (macOS, conda)

```bash
cd ~/Downloads/parkvision_ai
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests -q          # should end with: 11 passed
```

Notes:
* `ultralytics` installs PyTorch (a few hundred MB, needs internet once).
* `yolov8n-obb.pt` (~6 MB) downloads automatically the first time detection runs.
* YOLOv8 / Ultralytics is licensed **AGPL-3.0**, which is fine for a college project;
  commercial use needs an Ultralytics licence.
* On an Apple Silicon Mac, add `--device mps` to use the GPU.

---

## 4. Run it, in this order

| # | Command | What happens |
|---|---|---|
| 1 | Put a **fixed-camera, top-down** parking video in `data/` and set `VIDEO_PATH` in `config.py` | Input video (currently `data/parking2.mp4`, 960×540) |
| 2 | `python -m src.detector --device mps --no-show` | Detects vehicles on the first frame, saves `outputs/detection_test.jpg` |
| 3 | `python -m src.slot_marker --fresh` | Click 4 corners per slot, **s** to save, **q** to quit. For 4K video add `--region X Y W H` to zoom in |
| 4 | `python -m src.run_detection --device mps --loop` | Live annotated window (**q** quit, **p** pause); writes live status every 1 s, logs to database every 10 s |
| 5 | `python -m src.generate_synthetic` | Adds 8 weeks of **synthetic** history for ML testing |
| 6 | `python -m src.analysis --source synthetic` | Prints patterns, saves charts to `outputs/` |
| 7 | `python -m src.predict --source synthetic` | Trains, evaluates, saves model, prints 24 h forecast |
| 8 | `streamlit run dashboard.py` | Opens the dashboard in your browser (stop: **Ctrl + C**) |

Demo tip: run step 4 in one terminal and the dashboard in another. The
"Current occupancy" section refreshes itself every 2 seconds.

Useful options: `run_detection --source 0` (webcam), `--no-show`, `--no-db`, `--max-frames 300`.
Check the database at any time: `python -m src.database`.

---

## 5. How it works (viva notes)

**Object detection & YOLOv8n.** YOLO ("You Only Look Once") looks at the whole image once
and outputs bounding boxes, a class and a confidence score. The "n" (nano) version is the
smallest and fastest, good for a laptop.

**Why an aerial (OBB) model instead of the standard COCO model?** The standard
`yolov8n.pt` is trained on COCO, which shows cars mostly from the side at street level.
On a top-down drone view of ~300 parked cars it found only **4 cars** at the default
image size (640) and **17** at full 4K size. The aerial model `yolov8n-obb.pt` is trained
on the **DOTA** dataset (satellite and drone images) and has classes
`9 = large vehicle` and `10 = small vehicle`. On the same frame it found almost every car.
OBB models return rotated boxes in `result.obb`; the code uses their axis-aligned
`xyxy` box, so the rest of the pipeline did not need to change.

Two more detection settings matter for big aerial frames:
* `IMG_SIZE` (`imgsz`): YOLO resizes each frame before detecting. At 640, a car in a
  4K frame shrinks to ~10 px and is missed. Use 1920 for 4K and 1024 for 540p/720p.
* `MAX_DETECTIONS` (`max_det`): YOLO returns at most 300 boxes by default. A large lot
  can have more cars than that, so it is raised to 1000.

**Why mark slots manually?** YOLO finds vehicles, not parking spaces. Empty slots are just
painted lines, so their positions are marked once for a fixed camera and saved as polygons.
On a 4K frame the marker has to shrink the image 3× to fit a laptop screen, making each bay
~20 px wide; `--region X Y W H` shows one part of the frame at full size for accurate clicks.

**Occupancy decision.** For each vehicle, take its footprint (whole box for top-down views,
bottom half for angled CCTV, set by `FOOTPRINT_FRACTION` in `config.py`). Compute the overlap
area with every slot polygon (`cv2.intersectConvexConvex`). The vehicle is assigned only to
the slot with the highest score, and only if

`score = overlap area / min(slot area, footprint area) ≥ OVERLAP_THRESHOLD` (0.15).

Assigning each vehicle to one slot prevents a single car being counted twice.

**Smoothing.** A slot changes state only after 3 consecutive detection cycles agree,
which removes flicker caused by missed detections in a single frame.

**Live view vs history.** The latest frame and status are written every
`LIVE_UPDATE_SECONDS` (1 s) so the dashboard looks live, while the database gets only one
row every `LOG_INTERVAL_SECONDS` (10 s). The dashboard's current-status section uses
`@st.fragment(run_every="2s")`, so only that part re-runs instead of the whole page.

**Database design.** One table `occupancy_log(timestamp, total_slots, occupied, vacant,
occupancy_pct, source)`. `source` is `video`, `live` or `synthetic` so made-up data
can never be mixed with real observations.

**Prediction.** Target: average occupancy % in a future hour.
Features: hour, day of week, weekend flag, occupancy at the same hour yesterday (`lag_24`)
and the same hour last week (`lag_168`). Data is split **by time** (oldest 80 % train,
newest 20 % test). A random split would leak future information into training and give
falsely good scores. Two simple baselines (yesterday, last week) are compared with Linear
Regression and Random Forest using MAE and RMSE. Predictions are clipped to 0–100 % and
converted to available slots (never below 0 or above capacity). At least 14 days of
history are required.

**Results on synthetic data (8 weeks, 56 slots; test = last ~10 days):**

| Model | MAE | RMSE |
|---|---|---|
| Baseline: same hour yesterday | 5.78 | 9.87 |
| Baseline: same hour last week | 3.46 | 4.58 |
| Linear Regression | 3.36 | 4.49 |
| **Random Forest (chosen)** | **3.01** | **4.00** |

Errors are in percentage points of occupancy; 3 points ≈ 2 cars in a 56-slot lot. Synthetic
data is very regular, so the "last week" baseline is already strong. These numbers prove the
pipeline works; they are **not** real-world accuracy.

---

## 6. Development notes (problems found and how they were fixed)

| Problem | Cause | Fix |
|---|---|---|
| Almost no cars detected in the drone video | COCO model trained on street-level photos | Switched to aerial `yolov8n-obb.pt` (DOTA) |
| Cars missed in 4K frames | Default `imgsz=640` shrank cars to ~10 px | `IMG_SIZE` setting (1920 for 4K) |
| Detection count stuck at exactly 300 | YOLO's default `max_det=300` | `MAX_DETECTIONS = 1000` |
| Marked slots not on the bays (4K) | Bays only ~20 px wide in the shrunken marker window | `--region` zoom option in `slot_marker.py` |
| Slots drifted during the drone video | The drone camera moved; fixed slot coordinates need a fixed camera | Switched to a static top-down clip |
| Dashboard not updating | Page drew once; frame saved only every 10 s | `st.fragment(run_every="2s")` + separate 1 s live timer |
| Weekday chart showed equal bars | `st.bar_chart` mishandled text labels and the axis | Altair bar chart, fixed 0–100 axis, Mon→Sun order |

---

## 7. Limitations (be honest about these)

* Needs a **fixed camera**; if the camera moves (e.g. a drifting drone), slots no longer line up.
* **Occlusion**: cars hidden behind other cars or trees may be missed.
* **Shadows, night, rain, glare** reduce detection accuracy.
* The aerial model works best for top-down views; for angled CCTV the COCO model
  (`yolov8n.pt`, classes 2/3/5/7) with `FOOTPRINT_FRACTION = 0.5` may be better.
* A car parked across two slots is counted in one slot only.
* A recorded video is timestamped with the time it was **processed**, not when it was
  filmed, so a short video does not create real hourly history.
* The current test clip is a short (10 s) stock-footage preview with a watermark; it is
  used for development only and is not included in the repository.
* The ML forecast is only as good as the history it learns from. Real prediction needs
  weeks of real observations; synthetic results only prove the pipeline works.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Reinstall with `python -m pip install -r requirements.txt` |
| `No module named 'config'` | Run commands from the `parkvision_ai` folder, using `python -m src.xxx` |
| Video won't open | Check the path; convert unusual formats to `.mp4` (H.264) |
| Webcam not opening on Mac | System Settings → Privacy & Security → Camera → allow Terminal / VS Code |
| Ultralytics error mentioning `cv2` | Try `python -m pip install "opencv-python<5"` |
| Few cars detected in a top-down view | Use `yolov8n-obb.pt` and check `IMG_SIZE` |
| Detection slow | Add `--device mps`, lower `IMG_SIZE`, or increase `DETECT_EVERY_N_FRAMES` |
| Slots misaligned | Re-run `python -m src.slot_marker --fresh` (add `--region` for large frames) |
| Too many/few occupied | Tune `OVERLAP_THRESHOLD` and `FOOTPRINT_FRACTION` in `config.py` |
| Dashboard stuck on old data | Make sure `run_detection` is running; stop old servers with `pkill -f "streamlit run"` and restart |
