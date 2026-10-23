# ParkVision AI
**AI-Based Smart Parking Space Detection and Occupancy Prediction System Using Computer Vision and Machine Learning**

The system watches a parking area from a fixed camera, detects vehicles with YOLOv8n,
decides which marked parking slots are OCCUPIED or VACANT, logs occupancy to SQLite,
analyses historical patterns, forecasts occupancy for the next 24 hours, and shows
everything on a Streamlit dashboard.

---

## 1. Architecture

```
 Video / webcam / CCTV
        │  (OpenCV reads frames)
        ▼
 YOLOv8n vehicle detection ──► bounding boxes (car, motorcycle, bus, truck)
        │
        ▼
 Slot matching  ◄── data/slots.json  (marked once with slot_marker.py)
        │  each vehicle → the ONE slot it overlaps most
        ▼
 Smoothing (state must repeat 3 times) → OCCUPIED / VACANT per slot
        │
        ├──► Annotated video window (red = occupied, green = vacant)
        ├──► outputs/latest_status.json + latest_frame.jpg  (for dashboard)
        └──► SQLite data/parking.db  (one row every 10 s)
                    │
                    ▼
        Historical analysis (Pandas + Matplotlib)
                    │
                    ▼
        ML forecast (Scikit-learn: baseline vs Linear Regression vs Random Forest)
                    │
                    ▼
        Streamlit dashboard
```

**Current detection** (Computer Vision) and **future prediction** (Machine Learning)
are separate parts. CV answers "what is happening now?"; ML answers "what will
probably happen later?" based on stored history.

---

## 2. Folder structure

| Path | Purpose |
|---|---|
| `config.py` | All settings: paths, thresholds, intervals |
| `src/detector.py` | YOLOv8n wrapper; returns only vehicle boxes. Can test one image |
| `src/slot_marker.py` | Click-to-mark tool for parking slots → `data/slots.json` |
| `src/slots.py` | Load / save / validate / rescale slot coordinates |
| `src/occupancy.py` | Overlap logic, smoothing, counts, drawing |
| `src/run_detection.py` | Main pipeline: video → detection → occupancy → database |
| `src/database.py` | SQLite table, insert and read functions |
| `src/generate_synthetic.py` | Clearly-labelled synthetic history for testing ML |
| `src/analysis.py` | Data cleaning, hourly / weekday patterns, charts |
| `src/predict.py` | Feature preparation, time-ordered evaluation, forecast |
| `dashboard.py` | Streamlit dashboard |
| `tests/test_core.py` | 11 automatic tests (no YOLO or internet needed) |
| `data/` | Your video, `slots.json`, `parking.db` |
| `models/` | Saved prediction model |
| `outputs/` | Charts, latest frame and status |

---

## 3. Setup (macOS, conda environment `parkvision`)

```bash
conda activate parkvision
cd ~/Projects/parkvision_ai
python -m pip install -r requirements.txt
python -m tests.test_core          # should end with: 11/11 tests passed
```

Notes:
* `ultralytics` installs PyTorch (a few hundred MB, needs internet once).
* `yolov8n.pt` (~6 MB) downloads automatically the first time detection runs.
* YOLOv8 / Ultralytics is licensed **AGPL-3.0** — fine for a college project;
  commercial use needs an Ultralytics licence.
* On an Apple Silicon Mac, add `--device mps` to use the GPU.

---

## 4. Run it — in this order

| # | Command | What happens |
|---|---|---|
| 1 | Put a fixed-camera parking video at `data/parking.mp4` | Input video (e.g. from pexels.com, search "parking lot top view") |
| 2 | `python -m src.detector` | Detects vehicles on the first frame, saves `outputs/detection_test.jpg` |
| 3 | `python -m src.slot_marker` | Click 4 corners per slot, press **s** to save, **q** to quit |
| 4 | `python -m src.run_detection` | Live annotated window (**q** quit, **p** pause); logs to database |
| 5 | `python -m src.generate_synthetic` | Adds 8 weeks of **synthetic** history for ML testing |
| 6 | `python -m src.analysis --source synthetic` | Prints patterns, saves 4 charts to `outputs/` |
| 7 | `python -m src.predict --source synthetic` | Trains, evaluates, saves model, prints 24 h forecast |
| 8 | `streamlit run dashboard.py` | Opens the dashboard in your browser (stop: **Ctrl + C**) |

Tip for the demo: run step 4 with `--loop` in one terminal and the dashboard in another,
then press **Refresh** in the dashboard to see the status update.

Useful options: `run_detection --source 0` (webcam), `--no-show`, `--no-db`, `--max-frames 300`.
Check the database at any time: `python -m src.database`.

---

## 5. How it works (viva notes)

**Object detection & YOLOv8n.** YOLO ("You Only Look Once") looks at the whole image once
and outputs bounding boxes, a class (car, bus…) and a confidence score. The "n" (nano)
version is the smallest and fastest, good for a laptop. It is pretrained on the COCO
dataset, which already contains cars, motorcycles, buses and trucks (class IDs 2, 3, 5, 7),
so no training is needed.

**Why mark slots manually?** YOLO finds vehicles, not parking spaces. Empty slots are just
painted lines, so their positions are marked once for a fixed camera and saved as polygons.

**Occupancy decision.** For each vehicle, take its footprint (whole box for top-down views,
bottom half for angled CCTV — `FOOTPRINT_FRACTION` in `config.py`). Compute the overlap
area with every slot polygon (`cv2.intersectConvexConvex`). The vehicle is assigned only to
the slot with the highest score, and only if

`score = overlap area / min(slot area, footprint area) ≥ 0.30`.

Assigning each vehicle to one slot prevents a single car being counted twice.

**Smoothing.** A slot changes state only after 3 consecutive detection cycles agree,
which removes flicker caused by missed detections in a single frame.

**Database design.** One table `occupancy_log(timestamp, total_slots, occupied, vacant,
occupancy_pct, source)`. A row is saved every 10 seconds, not every frame, to avoid
thousands of duplicate rows. `source` is `video`, `live` or `synthetic` so made-up data
can never be mixed with real observations.

**Prediction.** Target: average occupancy % in a future hour.
Features: hour, day of week, weekend flag, occupancy at the same hour yesterday (`lag_24`)
and the same hour last week (`lag_168`). Data is split **by time** (oldest 80 % train,
newest 20 % test). A random split would leak future information into training and give
falsely good scores. Two simple baselines (yesterday, last week) are compared with Linear
Regression and Random Forest using MAE and RMSE. Predictions are clipped to 0–100 % and
converted to available slots (never below 0 or above capacity). At least 14 days of
history are required.

**Result on synthetic data (for reference only):** Random Forest MAE ≈ 3.2 percentage
points vs best baseline ≈ 3.7. These numbers describe made-up data, **not** real-world
accuracy.

---

## 6. Limitations (be honest about these)

* Needs a **fixed camera**; if the camera moves, slots must be re-marked.
* **Occlusion**: cars hidden behind other cars or trees may be missed.
* **Shadows, night, rain, glare** reduce YOLO accuracy.
* Very steep camera angles make bounding boxes cover neighbouring slots.
* A car parked across two slots is counted in one slot only.
* A recorded video is timestamped with the time it was **processed**, not when it was
  filmed, so a short video does not create real hourly history.
* The ML forecast is only as good as the history it learns from. Real prediction needs
  weeks of real observations; synthetic results only prove the pipeline works.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `conda activate parkvision`, then reinstall with `python -m pip install -r requirements.txt` |
| `No module named 'config'` | Run commands from the `parkvision_ai` folder, using `python -m src.xxx` |
| Video won't open | Check the path; convert unusual formats to `.mp4` (H.264) |
| Webcam not opening on Mac | System Settings → Privacy & Security → Camera → allow Terminal / VS Code |
| Ultralytics error mentioning `cv2` | Try `python -m pip install "opencv-python<5"` |
| Detection slow | Add `--device mps`, or increase `DETECT_EVERY_N_FRAMES` in `config.py` |
| Slots misaligned | Re-run `python -m src.slot_marker --fresh` |
| Too many/few occupied | Tune `OVERLAP_THRESHOLD` and `FOOTPRINT_FRACTION` in `config.py` |
