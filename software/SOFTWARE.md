# Drone rPPG — Software

Video acquisition and face detection software for rPPG (remote Photoplethysmography) signal extraction from a drone equipped with a XIAO ESP32-S3 camera.

---

## File Structure

```
software/
├── main.py                      # Entry point: selects mode, source, algorithm and display
├── DataHandler.py               # CameraHandler, WebcamHandler, IMUHandler
├── Processor.py                 # FaceProcessor: forehead ROI, RGB extraction, real-time HR (rPPG)
├── RespiratoryProcessor.py      # RespiratoryProcessor: pose tracking, real-time RR
├── ROIExtraction.py             # Forehead polygon (4 landmarks) with EMA smoothing
├── algoritmos/
│   ├── green.py                 # GREEN algorithm (Verkruysse 2008)
│   ├── omit.py                  # OMIT algorithm (Casado 2023)
│   ├── pos_wang.py              # POS algorithm (Wang et al. 2017)
│   └── adaptive_lms.py          # Adaptive NLMS filter with IMU (drone only)
└── test/
    ├── RecordPPG.py             # Record rPPG + hardware PPG simultaneously → CSV
    ├── RecordResp.py            # Record RR + hardware respiration sensor → CSV
    ├── ComputeMetrics.py        # Post-process raw CSVs → metrics CSVs in data_processed/
    ├── Evaluate.py              # Offline evaluation: MAE±SD per algorithm; plots to results/
    ├── RunTest.py               # Runs the full pipeline: ComputeMetrics → Evaluate
    ├── Arduino/
    │   └── SensorIntegration.ino    # Arduino: PPG + respiration sensors → Serial (115200 baud)
    ├── data/                    # Raw CSV recordings (timestamp + signal only)
    ├── data_processed/          # Processed CSVs with computed metrics
    └── results/                 # Evaluation plots (PNG)
```

---

## rPPG Pipeline

```
Frame (camera / webcam)
  → MediaPipe FaceMesh (face detection)
  → Forehead ROI (4 landmarks → forehead polygon → mean RGB/frame)
  → rgb_signal (N, 3)
  → [GREEN / OMIT / POS_WANG / LMS]  →  raw BVP (N,)
  → apply_filters  →  filtered BVP (N,)
  → estimate_hr  →  BPM  (find_peaks, 30 s window)
```

---

## Files

### `main.py`

Entry point. Prompts the user for the operating mode, display preference, video source and rPPG algorithm, then starts the main loop.

**Startup prompts (in order):**

| Step | Options | Description |
|------|---------|-------------|
| Mode | `1` rPPG · `2` Respiratory | `FaceProcessor` or `RespiratoryProcessor` |
| Display | `1` Yes · `2` No | Show matplotlib graphs + camera window, or terminal only |
| Source | `A` Drone · `B` Webcam | MJPEG stream or local webcam |
| Algorithm | `1`–`4` *(rPPG only)* | GREEN, OMIT, POS_WANG, LMS |

- **Display mode 2 (terminal only)** skips all `matplotlib` and `cv2.imshow` calls, significantly increasing FPS by eliminating rendering overhead.
- **Source A — Drone:** `CameraHandler` (MJPEG stream) + `IMUHandler` (polling `/imu`, rPPG mode only); fixed IP `http://192.168.4.1`
- **Source B — Webcam:** `WebcamHandler` (local OpenCV VideoCapture)

---

### `DataHandler.py`

Three data acquisition classes.

**Class: `CameraHandler`** — MJPEG stream from the drone camera

| Method | Description |
|--------|-------------|
| `__init__(base_url)` | Opens the connection and starts the capture thread |
| `update()` | Continuous thread that reads the MJPEG stream, extracts JPEG frames and stores the latest one |
| `get_frame()` | Returns the latest captured frame (`None` if not yet available) |
| `stop()` | Stops the capture thread |

**Technical details:**
- Stream read via `requests` with `stream=True`
- JPEG frames extracted from the binary buffer using `0xFF 0xD8` (start) and `0xFF 0xD9` (end) markers
- Separate daemon thread with automatic reconnection (2 s retry)
- Stream URL: `http://<IP>:81/stream`

**Class: `WebcamHandler`** — local webcam (PC)

| Method | Description |
|--------|-------------|
| `__init__()` | Opens `cv2.VideoCapture(0)` and starts capture thread |
| `get_frame()` | Returns the latest captured frame |
| `stop()` | Releases the VideoCapture |

**Class: `IMUHandler`** — drone IMU data via HTTP

| Method | Description |
|--------|-------------|
| `__init__(base_url)` | Starts a polling thread to the `/imu` endpoint at 10 Hz |
| `get_imu()` | Returns the latest JSON sample or `None` |
| `stop()` | Stops the thread |

JSON fields: `ax, ay, az` (g), `gx, gy, gz` (°/s), `bat` (mV), `fc_state`.

---

### `ROIExtraction.py`

Forehead ROI polygon extraction with EMA smoothing.

**Constant:**

| Constant | Value | Description |
|----------|-------|-------------|
| `ROI_FOREHEAD` | `"FOREHEAD"` | 4-landmark forehead quadrilateral |

**Landmark indices:**

- `FOREHEAD_ROI = [103, 332, 296, 66]` — 4 forehead landmarks (MediaPipe FaceMesh)

**Function:**

| Function | Description |
|----------|-------------|
| `get_roi_polygon(landmarks, w, h, ema, alpha)` | Returns forehead polygon with EMA smoothing (`alpha=0.35`) |

---

### `Processor.py`

Processes frames with MediaPipe FaceMesh, extracts the forehead RGB signal, runs rPPG algorithms and estimates HR in real time.

**Class: `FaceProcessor`**

| Method | Description |
|--------|-------------|
| `__init__(camera, imu, algo, display)` | Initializes FaceMesh, signal buffers, background threads; `display=False` disables all rendering |
| `process_frame(frame)` | Resizes to 320×240, runs FaceMesh, extracts forehead ROI RGB, draws overlay |
| `get_fps(window=60)` | Estimates FPS from real timestamps; fallback 27.0 Hz |
| `apply_filters(bvp, fs)` | Detrend + Butterworth bandpass [0.75–4.0 Hz] |
| `estimate_hr(bvp, fs)` | `find_peaks` on filtered BVP → mean RRI → BPM |
| `_estimate_hr_realtime()` | Computes BVP + HR on last 30 s of signal |
| `_compute_hr_background()` | Background thread: HR every 30 frames |
| `run()` | Main acquisition + display loop |
| `stop()` | Closes FaceMesh and camera |

**FaceMesh configuration:**
- `max_num_faces=1`, `refine_landmarks=False`
- `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`

**Filters (`apply_filters`):**

1. **Detrend** (λ=100) — removes slow baseline drift (Tarvainen sparse method)
2. **Butterworth bandpass** [0.75–4.0 Hz], order 2, `filtfilt` — cardiac band (45–240 bpm)

**HR estimation (`estimate_hr`):**

`find_peaks` with `distance=0.5 s` and `prominence=0.25×std`; RRI filter [0.4–1.5 s]; HR = 60 / mean(RRI). Consistent with `ComputeMetrics.metrics_from_peaks`.

**Background thread trigger:**

| Trigger | Interval | Task |
|---------|----------|------|
| Every 30 frames | ~2 s @ 15 fps | HR estimation |

---

### `RespiratoryProcessor.py`

Estimates respiratory rate (RR) in real time using the **Bartula (2013)** camera-based algorithm. Uses MediaPipe Pose **once at startup** to locate the chest and derive a tight ROI; Pose is then closed and the Bartula algorithm runs on that fixed ROI for the rest of the session.

> Bartula, M., Tigges, T., & Muehlsteff, J. *Camera-based System for Contactless Monitoring of Respiration.* IEEE EMBS, 2013.

**Class: `RespiratoryProcessor`**

| Method | Description |
|--------|-------------|
| `__init__(camera, display)` | Initialises signal buffers, position integrator and background thread state |
| `_init_roi_from_pose(timeout)` | Runs Pose on live frames until shoulders detected; computes chest ROI; closes Pose |
| `process_frame(frame)` | Extracts profile, computes shift, integrates position, detects global motion |
| `get_fps(window=60)` | Estimates FPS from recent timestamps; fallback 25.0 Hz |
| `_make_profile(roi_gray)` | 1D vertical profile: mean+std per row, high-pass filtered |
| `_profile_shift(p_curr, p_prev)` | Phase cross-correlation with Hann window; sub-pixel quadratic interpolation |
| `_global_motion(curr, prev)` | Block-based frame-difference motion detector |
| `apply_filters(sig, fs)` | Linear detrend + Butterworth bandpass [0.1–0.5 Hz] |
| `_estimate_rr_from_peaks(filtered, fs, flags)` | Peak detection + breath-by-breath validation |
| `_compute_rr_background()` | Background thread: RR every 30 frames |
| `run()` | Calls `_init_roi_from_pose`, then main acquisition + display loop |
| `stop()` | Stops camera and closes windows |

**Bartula pipeline (per frame):**

```
Frame ROI (grayscale)
  → _make_profile       : mean(row) + std(row) → high-pass → 1D vector (H,)
  → _profile_shift      : Hann window → FFT cross-correlation → sub-pixel peak
                          r = F⁻¹( F(pₜ) · conj(F(pₜ₋₁)) )
  → integrate shift     : position += shift  (chest displacement signal)
  → _global_motion      : block diff > MOTION_THRESH → flag frame
  → apply_filters       : detrend + Butterworth [0.1–0.5 Hz]
  → _estimate_rr_from_peaks : find_peaks → validate each breath interval
                              (duration ∈ [1.5, 10] s, motion_ratio < 0.5)
  → median(valid intervals) → rpm → EMA → rr_estimate
```

**Parameters:**

| Constant | Value | Description |
|----------|-------|-------------|
| `ROI_PAD_X_FACTOR` | 0.15 | Horizontal ROI padding as fraction of shoulder width |
| `ROI_HEIGHT_FACTOR` | 1.20 | Estimated chest height as fraction of shoulder width (no hips) |
| `ROI_TOP_FACTOR` | 0.10 | ROI top margin above shoulder line |
| `HP_KERNEL` | 21 | Moving-average kernel for profile high-pass |
| `BLOCK_SIZE` | 16 px | Block size for motion detector |
| `MOTION_THRESH` | 12 | Mean abs diff threshold per block |
| `GLOBAL_MOTION_RATIO` | 0.30 | Moving-block fraction to flag global motion |
| `MIN_SEC` | 20 s | Minimum signal window before first estimate |
| `RR_ALPHA` | 0.15 | EMA smoothing factor |
| `MIN_BREATH_S` | 1.5 s | Shortest valid breath (≈ 40 rpm) |
| `MAX_BREATH_S` | 10.0 s | Longest valid breath (6 rpm) |

---

### `test/`

Hardware-synchronized recording and evaluation scripts.

**`Arduino/SensorIntegration.ino`** — Arduino sketch that reads an analog PPG sensor (pin `A6`) and a respiration sensor (pin `A5`) and streams timestamped samples over Serial at 115200 baud.

---

**`RecordPPG.py`** — Records rPPG and hardware PPG simultaneously and saves to CSV.

- **Warmup:** 30 s (no data saved); **Recording:** 60 s
- Outputs to `data/`:
  - `BVP_<ALGO>_<ts>_sw.csv` — `timestamp`, `signal_sw` (z-normalised BVP)
  - `BVP_<ALGO>_<ts>_hw.csv` — `timestamp`, `signal_hw` (raw PPG, relative time)
- HW signal read via Serial (Arduino, `COM3`, 115200 baud)

---

**`RecordResp.py`** — Records respiratory rate and hardware respiration sensor simultaneously and saves to CSV.

- **Warmup:** 20 s; **Recording:** 60 s
- Outputs to `data/`:
  - `RESP_<ts>_sw.csv` — `timestamp`, `signal_sw`
  - `RESP_<ts>_hw.csv` — `timestamp`, `signal_hw`

---

**`ComputeMetrics.py`** — Post-processes raw CSVs from `data/` into metric CSVs in `data_processed/`.

```bash
python ComputeMetrics.py                             # all pairs in data/
python ComputeMetrics.py --file BVP_GREEN_X_sw.csv  # single pair
```

**rPPG processing (`process_pair`):**
- SW: cumulative expanding window; `metrics_from_peaks` → HR, SDNN, RMSSD per timestamp
- HW: same, with bandpass filter [0.75–4.0 Hz] applied first
- Outputs:
  - `BVP_<ALGO>_<ts>_sw_processed.csv` — `timestamp`, `signal_sw`, `HR_<algo>`, `SDNN_<algo>`, `RMSSD_<algo>`
  - `BVP_<ALGO>_<ts>_hw_processed.csv` — `timestamp`, `signal_hw`, `HR_gt`, `SDNN_gt`, `RMSSD_gt`

**Respiratory processing (`process_resp_pair`):**
- SW downsampled to 25 Hz before loop (avoids O(n²) at high acquisition rates)
- Sliding 30 s window; `resp_metrics_from_peaks` → RR, BBI per timestamp
- Outputs:
  - `RESP_<ts>_sw_processed.csv` — `timestamp`, `signal_sw`, `RR_BARTULA`, `BBI_BARTULA`
  - `RESP_<ts>_hw_processed.csv` — `timestamp`, `signal_hw`, `RR_gt`, `BBI_gt`

**Peak detection (`metrics_from_peaks`):**
- `find_peaks` with `distance=0.5 s`, `prominence=0.25×std`
- RRI filter: [0.4–1.5 s]; HR = 60 / mean(RRI); SDNN = std(RRI)×1000; RMSSD = √mean(ΔRRI²)×1000

**Peak detection (`resp_metrics_from_peaks`):**
- `find_peaks` with `distance=1.5 s`, `prominence=0.25×std`
- BBI filter: [1.5–10.0 s]; RR = 60 / mean(BBI)

---

**`Evaluate.py`** — Offline evaluation. Reads processed CSVs from `data_processed/`, interpolates HW ground truth onto SW timestamps, and computes MAE±SD per algorithm.

```bash
python Evaluate.py               # all recordings in data_processed/
python Evaluate.py --file BVP_GREEN_X_sw_processed.csv  # single file
```

**Warmup cutoff:**

| Recording type | Cutoff | Reason |
|----------------|--------|--------|
| rPPG (HR_*) | 30 s | Matches the 30 s minimum window in `FaceProcessor._estimate_hr_realtime` |
| Respiratory (RR_BARTULA) | 20 s | Matches `MIN_SEC = 20` in `RespiratoryProcessor` |

**GT interpolation:** HW metrics (on HW timestamps) are linearly interpolated onto SW timestamps before computing error metrics.

**Metrics:** MAE ± SD only.

**Output — three tables:**

| Table | Content |
|-------|---------|
| Table 1 — HR | MAE±SD per algorithm (GREEN, OMIT, POS_WANG) |
| Table 2 — RR | MAE±SD for BARTULA |
| Table 3 — HRV | SDNN and RMSSD MAE±SD per algorithm |

**Plots:** per-recording signal + metrics overlay (SW vs HW), cropped to 0–60 s; saved to `results/`.

---

**`RunTest.py`** — Runs the full pipeline in order:

```bash
python RunTest.py
```

1. `ComputeMetrics.py` → `data_processed/`
2. `Evaluate.py` → prints tables + saves plots to `results/`

---

**Common configuration** (top of each recording script):

| Constant | Default | Description |
|----------|---------|-------------|
| `SERIAL_PORT` | `COM3` | Serial port of the Arduino |
| `BAUD_RATE` | `115200` | Must match `SensorIntegration.ino` |
| `XIAO_IP` | `http://192.168.4.1` | Drone IP (ignored when using webcam) |
| `WARMUP_S` | 30 (PPG) / 20 (Resp) | Warmup duration before recording starts |
| `RECORD_S` | 60 | Recording duration |

---

### `algoritmos/`

Each algorithm receives `rgb (N, 3)` (and optionally `fs` or `imu`) and returns `bvp (N,)` — raw BVP signal, without filters.

#### `green.py` — GREEN

> Verkruysse, W., Svaasand, L. O. & Nelson, J. S. *Remote plethysmographic imaging using ambient light.* Optical Express 16, 21434–21445 (2008).

Extracts the green channel directly. Hemoglobin absorbs strongly in the green band (~550 nm), making it the channel with the highest pulse amplitude.

```
BVP = G
```

---

#### `omit.py` — OMIT (Orthogonal Matrix Image Transformation)

> Álvarez Casado, C., & Bordallo López, M. *Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.* IEEE JBHI (2023).

Uses QR decomposition to remove the dominant component of the RGB signal and extract the pulse in the orthogonal subspace. Robust to video compression artifacts.

```
A = rgb.T                    # (3, N)
Q, R = qr(A)
S = Q[:, 0]                  # dominant direction
P = I - Sᵀ·S                 # orthogonal projector
Y = P @ A
BVP = Y[1, :]
```

---

#### `pos_wang.py` — POS (Plane-Orthogonal-to-Skin)

> Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. *Algorithmic principles of remote PPG.* IEEE TBME, 64(7), 1479–1491 (2017).

1.6 s sliding window. Temporally normalizes RGB and projects onto a plane orthogonal to the skin tone to separate pulse from intensity variations.

```
l = ceil(1.6 × fs)
Cn = RGB[m:n] / mean(RGB[m:n])
S = [[0,1,-1],[-2,1,1]] @ Cn
h = S[0] + σ(S[0])/σ(S[1]) × S[1]
H[m:n] += h - mean(h)           # overlap-add
```

---

#### `adaptive_lms.py` — Adaptive LMS with IMU

> Widrow, B. & Hoff, M. E. *Adaptive switching circuits.* IRE WESCON (1960).

Cancels motion artifacts using the IMU signal as noise reference. NLMS filtering adapts its weights to estimate and subtract the motion component from the green signal.

```
green = rgb[:, 1]
imu_ref = interp(IMU, N)
error[n] = green[n] - w·x[n]
w += (μ / (||x||² + ε)) × error × x   # NLMS weight update
```

**Parameters:** `μ=0.1`, `ε=1e-3`. Drone mode only.

---

## Dependencies

```
opencv-python
mediapipe
requests
numpy
scipy
matplotlib
pandas
```

Install:
```bash
pip install opencv-python mediapipe requests numpy scipy matplotlib pandas
```

> Requires **Python 3.11** (MediaPipe has limited compatibility with newer versions)

---

## How to Run

```bash
python main.py
```

The program prompts:
1. Mode: `1` rPPG · `2` Respiratory
2. Display: `1` graphs + camera · `2` terminal only (higher FPS)
3. Video source: `A` (drone, `192.168.4.1`) or `B` (PC webcam)
4. *(rPPG only)* Algorithm: `1` GREEN · `2` OMIT · `3` POS_WANG · `4` LMS

Press `q` to stop.

### Record and evaluate

```bash
# Record (run from test/)
python RecordPPG.py    # rPPG + hardware PPG  → data/BVP_<ALGO>_<ts>_{sw,hw}.csv
python RecordResp.py   # RR  + hardware resp  → data/RESP_<ts>_{sw,hw}.csv

# Evaluate all recordings
python RunTest.py
```

---

## Status and Next Steps

- [x] MJPEG stream from XIAO ESP32 camera
- [x] Face detection with MediaPipe FaceMesh (468 landmarks)
- [x] Forehead ROI (4 landmarks, EMA-smoothed polygon)
- [x] GREEN, OMIT, POS_WANG and LMS algorithms
- [x] Filters: detrend + Butterworth bandpass [0.75–4.0 Hz]
- [x] Real-time HR via find_peaks (every 30 frames, 30 s window)
- [x] LMS motion artifact cancellation with IMU
- [x] Respiratory rate via MediaPipe Pose (thorax movement, 0.1–0.5 Hz)
- [x] Display toggle: full visual mode vs terminal-only (improved FPS)
- [x] Hardware-synchronized recording scripts (RecordPPG, RecordResp)
- [x] ComputeMetrics: peak-based HR/HRV/RR metrics → data_processed/
- [x] Evaluate: MAE±SD per algorithm, GT interpolation, plots to results/
- [x] Consistent peak-based HR estimation across real-time and offline evaluation
- [ ] IMU-based motion compensation for GREEN / OMIT / POS
- [ ] Clinical validation against oximeter ground truth
