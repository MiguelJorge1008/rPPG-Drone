# Drone rPPG — Software

Video acquisition and face detection software for rPPG (remote Photoplethysmography) signal extraction from a drone equipped with a XIAO ESP32-S3 camera.

---

## File Structure

```
software/
├── main.py                      # Entry point: selects mode, source, algorithm, ROI and display
├── DataHandler.py               # CameraHandler, WebcamHandler, IMUHandler
├── Processor.py                 # FaceProcessor: ROI, RGB extraction, real-time HR (rPPG)
├── RespiratoryProcessor.py      # RespiratoryProcessor: pose tracking, real-time RR
├── ROIExtraction.py             # ROI modes, face polygon, grid extraction, DMRS
├── evaluate.py                  # Offline evaluation: MAE, RMSE, PCC from CSV
├── algoritmos/
│   ├── green.py                 # GREEN algorithm (Verkruysse 2008)
│   ├── omit.py                  # OMIT algorithm (Casado & López 2023)
│   ├── pos_wang.py              # POS algorithm (Wang et al. 2017)
│   └── adaptive_lms.py          # Adaptive NLMS filter with IMU (drone only)
└── validation/
    ├── SensorIntegration.ino        # Arduino: PPG + respiration sensors → Serial (115200 baud)
    ├── PPG2csv.py                   # Record rPPG + hardware PPG simultaneously → CSV
    ├── Resp2csv.py                  # Record RR + hardware respiration sensor → CSV
    ├── Validation_rPPG_PPG.py       # Live rPPG vs hardware PPG comparison
    ├── Validation_Resp_Resp.py      # Live RR vs hardware respiration comparison
    └── Validation_rPPG_SMARTLOCK.py # rPPG vs SMARTLOCK reference device
```

---

## rPPG Pipeline

```
Frame (camera / webcam)
  → MediaPipe FaceMesh (face detection)
  → ROI extraction (FOREHEAD / FACE / MULTI)
       FOREHEAD : 4 landmarks → forehead polygon → mean RGB/frame
       FACE     : 36 landmarks → face oval polygon → mean RGB/frame
       MULTI    : face oval → 9×9 grid → DMRS region selection → mean RGB/frame
  → rgb_signal (N, 3)
  → [GREEN / OMIT / POS_WANG / LMS]  →  raw BVP (N,)
  → apply_filters  →  filtered BVP (N,)
  → estimate_hr  →  BPM  (FFT, range 45–240 bpm)
```

---

## Files

### `main.py`

Entry point. Prompts the user for the operating mode, display preference, video source, rPPG algorithm and ROI mode, then starts the main loop.

**Startup prompts (in order):**

| Step | Options | Description |
|------|---------|-------------|
| Mode | `1` rPPG · `2` Respiratory | `FaceProcessor` or `RespiratoryProcessor` |
| Display | `1` Sim · `2` Não | Show matplotlib graphs + camera window, or terminal only |
| Source | `A` Drone · `B` Webcam | MJPEG stream or local webcam |
| Algorithm | `1`–`4` *(rPPG only)* | GREEN, OMIT, POS_WANG, LMS |
| ROI | `1`–`3` *(rPPG only)* | FOREHEAD, FACE, MULTI |

- **Display mode 2 (terminal only)** skips all `matplotlib` and `cv2.imshow` calls, significantly increasing FPS by eliminating rendering overhead.
- **Source A — Drone:** `CameraHandler` (MJPEG stream) + `IMUHandler` (polling `/imu`, rPPG mode only); fixed IP `http://192.168.4.1`
- **Source B — Webcam:** `WebcamHandler` (local OpenCV VideoCapture)
- **Available algorithms (rPPG only):** GREEN, OMIT, POS_WANG, LMS (LMS only in drone mode)
- **ROI modes (rPPG only):** FOREHEAD, FACE, MULTI

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

All ROI logic: landmark indices, polygon extraction, grid extraction and DMRS region selection.

**ROI mode constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `ROI_FOREHEAD` | `"FOREHEAD"` | 4-landmark forehead quadrilateral (original) |
| `ROI_FACE` | `"FACE"` | 36-landmark face oval contour |
| `ROI_MULTI` | `"MULTI"` | Face oval + 9×9 grid + DMRS selection (Face2PPG) |

**Landmark indices:**

- `FOREHEAD_ROI = [103, 332, 296, 66]` — 4 forehead landmarks
- `FACE_OVAL` — 36 ordered landmarks tracing the face contour

**Functions:**

| Function | Description |
|----------|-------------|
| `get_roi_polygon(landmarks, w, h, ema, alpha, roi_mode)` | Returns face polygon with EMA smoothing; dispatches to FOREHEAD or FACE_OVAL indices |
| `extract_grid_rgb(rgb, face_mask, poly, h, w, face_mean)` | Divides face bounding box into GRID_N×GRID_N cells; returns mean RGB per cell |
| `draw_grid_overlay(frame, poly, h, w, selected_indices)` | Draws grid on frame; selected regions shown in green |
| `compute_kfd(signal)` | Katz Fractal Dimension — measures signal complexity |
| `compute_dfa(signal)` | Detrended Fluctuation Analysis — measures long-range correlations |
| `dmrs_select(region_signals, fs, r_max, kfd_thresh)` | Dynamic Multi-Region Selection: variance → KFD → spectral energy ranking |

**DMRS pipeline (Face2PPG, Casado & López 2023):**

1. Discard regions with zero variance
2. KFD filter: keep regions with `KFD_i / KFD_global >= 0.85`
3. Rank by spectral energy in HR band [0.75–4.0 Hz] → top `R_MAX=32` regions

> DFA omitted from real-time DMRS for performance. DMRS runs in a background thread every 90 frames; HR estimation uses the cached selected regions every 30 frames.

**Grid parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GRID_N` | 9 | Grid rows/columns (9×9 = 81 regions) |
| `R_MAX` | 32 | Max regions selected by DMRS |
| `MIN_REGION_PIXELS` | 5 | Min pixels inside face mask for a cell to be valid |

---

### `Processor.py`

Processes frames with MediaPipe FaceMesh, extracts the RGB signal, runs rPPG algorithms and estimates HR in real time.

**Class: `FaceProcessor`**

| Method | Description |
|--------|-------------|
| `__init__(camera, imu, algo, roi, display)` | Initializes FaceMesh, signal buffers, background threads; `display=False` disables all rendering |
| `process_frame(frame)` | Resizes to 320×240, runs FaceMesh, extracts ROI RGB, draws overlay |
| `get_fps(window=60)` | Estimates FPS from real timestamps; fallback 27.0 Hz |
| `apply_filters(bvp, fs)` | Detrend + Butterworth bandpass [0.75–4.0 Hz] |
| `estimate_hr(bvp, fs)` | FFT peak in [0.75–4.0 Hz] → BPM |
| `_estimate_hr_realtime()` | Computes BVP + HR on signal window; uses DMRS regions if ROI_MULTI |
| `_compute_hr_background()` | Background thread: HR every 30 frames |
| `_compute_dmrs_background()` | Background thread: DMRS region selection every 90 frames (ROI_MULTI only) |
| `run()` | Main acquisition + display loop |
| `stop()` | Closes FaceMesh and camera |

**FaceMesh configuration:**
- `max_num_faces=1`, `refine_landmarks=False`
- `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`

**Filters (`apply_filters`):**

1. **Detrend** (λ=100) — removes slow baseline drift (Tarvainen sparse method)
2. **Butterworth bandpass** [0.75–4.0 Hz], order 2, `filtfilt` — cardiac band (45–240 bpm)

**HR estimation (`estimate_hr`):**

FFT over the filtered BVP signal; peak frequency in [0.75–4.0 Hz] converted to BPM.

**Background thread triggers:**

| Trigger | Interval | Task |
|---------|----------|------|
| Every 30 frames | ~2 s @ 15 fps | HR estimation (lightweight) |
| Every 90 frames | ~6 s @ 15 fps | DMRS region selection (ROI_MULTI only) |

---

### `RespiratoryProcessor.py`

Estimates respiratory rate (RR) in real time using the **Bartula (2013)** camera-based algorithm. Uses MediaPipe Pose **once at startup** to locate the chest and derive a tight ROI; Pose is then closed and the Bartula algorithm runs on that fixed ROI for the rest of the session.

> Bartula, M., Tigges, T., & Muehlsteff, J. *Camera-based System for Contactless Monitoring of Respiration.* IEEE EMBS, 2013.

**Class: `RespiratoryProcessor`**

| Method | Description |
|--------|-------------|
| `__init__(camera, display)` | Initialises signal buffers, position integrator and background thread state; `display=False` disables all rendering |
| `_init_roi_from_pose(timeout)` | Runs Pose on live frames until shoulders detected; computes chest ROI; closes Pose. Raises `RuntimeError` on timeout. |
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

**Startup — ROI detection:**

```
Live frames → MediaPipe Pose
  shoulders visible (visibility > 0.5)?
    YES → compute chest ROI from shoulder landmarks → show 1 s → close Pose → start Bartula
    NO  → keep waiting … timeout (15 s) → RuntimeError
```

ROI boundaries derived from shoulder landmarks:

| Edge | Formula |
|------|---------|
| Top | `shoulder_y − 10 % × shoulder_width` |
| Bottom | `hip_y` if hips visible, else `shoulder_y + 120 % × shoulder_width` |
| Left / Right | `shoulder edges ± 15 % × shoulder_width` |

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

> The ROI rectangle is drawn in yellow (normal) or red (global motion detected). If Pose does not detect the subject within 15 s the program raises an error — no measurement without a valid chest ROI.

---

### `validation/`

Hardware-synchronized validation scripts. All scripts read a hardware reference sensor via Serial (Arduino at 115200 baud, `COM5`) and synchronize it with the software pipeline in real time.

**`SensorIntegration.ino`** — Arduino sketch that reads an analog PPG sensor (pin `A6`) and a respiration sensor (pin `A5`) and streams timestamped samples over Serial at 115200 baud.

---

**`PPG2csv.py`** — Records rPPG and hardware PPG simultaneously and saves to CSV for offline evaluation.

- Reads hardware PPG from Serial while running `FaceProcessor`
- Saves synchronized timestamps, hardware PPG raw values and rPPG HR estimates
- Plots both signals in real time

**`Resp2csv.py`** — Records respiratory rate and hardware respiration sensor simultaneously and saves to CSV.

- Reads hardware respiration signal from Serial while running `RespiratoryProcessor`
- Saves synchronized timestamps and RR estimates

---

**`Validation_rPPG_PPG.py`** — Live side-by-side comparison of rPPG vs hardware PPG reference.

- Runs `FaceProcessor` and reads hardware PPG from Serial in parallel
- Displays both BVP signals and HR estimates in real time

**`Validation_Resp_Resp.py`** — Live comparison of software RR vs hardware respiration sensor.

- Runs `RespiratoryProcessor` and reads hardware respiration from Serial in parallel
- Displays both position signals and RR estimates side by side

**`Validation_rPPG_SMARTLOCK.py`** — Compares rPPG HR estimates against a SMARTLOCK medical reference device via Serial.

---

**Common configuration** (top of each validation script):

| Constant | Default | Description |
|----------|---------|-------------|
| `SERIAL_PORT` | `COM5` | Serial port of the Arduino |
| `BAUD_RATE` | `115200` | Must match `SensorIntegration.ino` |
| `XIAO_IP` | `http://192.168.4.1` | Drone IP (ignored when using webcam) |

---

### `evaluate.py`

Offline evaluation script. Reads a results CSV and computes MAE, RMSE and PCC per algorithm, with optional breakdown by ROI mode and source.

**Expected CSV columns:**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Time of measurement |
| `HR_gt` | float | Ground truth HR in BPM (oximeter) |
| `HR_GREEN` | float | Estimated HR — GREEN (optional) |
| `HR_OMIT` | float | Estimated HR — OMIT (optional) |
| `HR_POS_WANG` | float | Estimated HR — POS_WANG (optional) |
| `HR_LMS` | float | Estimated HR — LMS (optional) |
| `roi_mode` | str | FOREHEAD / FACE / MULTI (optional) |
| `source` | str | drone / webcam (optional) |

**Usage:**

```bash
python evaluate.py results.csv           # MAE / RMSE / PCC tables
python evaluate.py results.csv --plot    # + scatter plots per algorithm
python evaluate.py results.csv --save    # + save metrics to results_metrics.csv
```

**Output:** tables with MAE (BPM), RMSE (BPM) and PCC broken down by algorithm, ROI mode and source — equivalent to Face2PPG Table I format.

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

**Parameters:** `μ=0.01`, `ε=1e-6`. Drone mode only.

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
5. *(rPPG only)* ROI mode: `1` Forehead · `2` Full face · `3` Multi-region (DMRS)

Press `q` to stop.

### Evaluate results

```bash
python evaluate.py results.csv --plot --save
```

---

## Status and Next Steps

- [x] MJPEG stream from XIAO ESP32 camera
- [x] Face detection with MediaPipe FaceMesh (468 landmarks)
- [x] ROI modes: Forehead (4 landmarks) / Face oval (36 landmarks) / Multi-region DMRS
- [x] 9×9 grid extraction with DMRS region selection (Face2PPG)
- [x] Grid overlay visualisation in camera window
- [x] GREEN, OMIT, POS_WANG and LMS algorithms
- [x] Filters: detrend + Butterworth bandpass [0.75–4.0 Hz]
- [x] Real-time HR via FFT (every 30 frames)
- [x] LMS motion artifact cancellation with IMU
- [x] Evaluation script: MAE / RMSE / PCC from CSV
- [x] Respiratory rate via MediaPipe Pose (thorax movement, 0.1–0.5 Hz)
- [x] Display toggle: full visual mode vs terminal-only (improved FPS)
- [x] Hardware-synchronized validation scripts (PPG, respiration, SMARTLOCK)
- [ ] Peak detection → RR intervals → HRV (SDNN, RMSSD, LF, HF, LF/HF)
- [ ] IMU-based motion compensation for GREEN / OMIT / POS
- [ ] Clinical validation against oximeter ground truth
