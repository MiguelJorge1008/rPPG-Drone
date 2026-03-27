# Drone rPPG — Software

Video acquisition and face detection software for rPPG (remote Photoplethysmography) signal extraction from a drone equipped with a XIAO ESP32-S3 camera.

---

## File Structure

```
software/
├── main.py                  # Entry point: selects video source and algorithm
├── DataHandler.py           # CameraHandler, WebcamHandler, IMUHandler
├── Processor.py             # Face detection, ROI, filters, real-time HR
├── algoritmos/
│   ├── green.py             # GREEN algorithm (Verkruysse 2008)
│   ├── omit.py              # OMIT algorithm (Casado & López 2023)
│   ├── pos_wang.py          # POS algorithm (Wang et al. 2017)
│   └── adaptive_lms.py      # Adaptive NLMS filter with IMU (drone only)
└── SOFTWARE.md
```

---

## rPPG Pipeline

```
Frame (camera / webcam)
  → MediaPipe FaceMesh (face detection)
  → Forehead ROI (4 landmarks) → mean RGB per frame
  → rgb_signal  (N, 3)
  → [GREEN / OMIT / POS_WANG / LMS]  →  raw BVP  (N,)
  → apply_filters  →  filtered BVP  (N,)
  → estimate_hr  →  BPM  (FFT, range 45–240 bpm)
```

---

## Files

### `main.py`

Entry point. Prompts the user for the video source and rPPG algorithm, instantiates the handlers and starts the main loop.

- **Source A — Drone:** `CameraHandler` (MJPEG stream) + `IMUHandler` (polling `/imu`); fixed IP `http://192.168.4.1`
- **Source B — Webcam:** `WebcamHandler` (local OpenCV VideoCapture)
- **Available algorithms:** GREEN, OMIT, POS_WANG, LMS (LMS only available in drone mode, requires IMU)

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

### `Processor.py`

Processes frames with MediaPipe FaceMesh, extracts the forehead RGB signal, runs rPPG algorithms and filters the BVP signals.

**Class: `FaceProcessor`**

| Method | Description |
|--------|-------------|
| `__init__(camera, algorithm, imu)` | Initializes FaceMesh, `rgb_signal`, timestamps and live RGB plot |
| `process_frame(frame)` | Resizes frame, runs FaceMesh, extracts ROI RGB, draws landmarks |
| `get_forehead_polygon(landmarks, w, h)` | Converts 4 forehead landmarks to pixel coordinates |
| `get_fps(window=60)` | Estimates FPS from real timestamps of the last `window` frames; fallback 27.0 Hz |
| `apply_filters(bvp, fs)` | Detrend + Butterworth bandpass [0.75–4.0 Hz] on the BVP signal |
| `_update_plot()` | Updates the live RGB plot (sliding window of 300 frames) |
| `run()` | Main loop: capture frames, process, display video and update plot every 5 frames |
| `stop()` | Closes FaceMesh, stops camera, runs selected algorithm and shows BVP plot + BPM |

**FaceMesh configuration:**
- `max_num_faces=1` — detects only one face
- `refine_landmarks=False` — no iris refinement
- `min_detection_confidence=0.5`
- `min_tracking_confidence=0.5`

**Performance optimizations:**
- Frame resized to **320×240** before MediaPipe
- `rgb.flags.writeable = False` before `process()` (MediaPipe optimization)
- Only landmark points drawn, no connections (`connections=None`)

**Forehead ROI:**

Central quadrilateral defined by 4 landmarks: `[103, 332, 296, 66]`

| Index | Position |
|-------|----------|
| 103 | Top left |
| 332 | Top right |
| 296 | Bottom right (above eyebrow) |
| 66 | Bottom left (above eyebrow) |

Yellow outline with semi-transparent fill (20% opacity).

**RGB signal (`rgb_signal`):**
- Per frame with detected face: all pixels inside the ROI polygon → spatial mean → `[R, G, B]`
- Accumulated in `self.rgb_signal`; converted to `np.ndarray (N, 3)` on `stop()`

**FPS estimation (`get_fps`):**
- Calculated from real frame timestamps (`time.time()`)
- Avoids relying on the nominal camera FPS, which can vary in MJPEG streams
- Fallback: `27.0 Hz`

**Filters (`apply_filters`):**

Applied in the `Processor` after each algorithm, not inside the algorithms. Following Face2PPG (Casado & López, 2023) and Wang et al. (2017), which describes the POS core without filtering.

1. **Detrend** (λ=100) — removes slow baseline drift
2. **Butterworth bandpass** [0.75–4.0 Hz], order 2, `filtfilt` — cardiac band (45–240 bpm)

**Output on `stop()`:**

Matplotlib figure with the filtered BVP from the selected algorithm and the estimated BPM in the title. The real `fs` (measured at runtime) is indicated.

---

### `algoritmos/`

Each algorithm receives `rgb (N, 3)` (and optionally `fs` or `imu`) and returns `bvp (N,)` — raw BVP signal, without filters.

#### `green.py` — GREEN

> Verkruysse, W., Svaasand, L. O. & Nelson, J. S. *Remote plethysmographic imaging using ambient light.* Optical Express 16, 21434–21445 (2008).

Extracts the green channel directly. Hemoglobin absorbs strongly in the green band (~550 nm), making it the channel with the highest pulse amplitude.

```
BVP = G
```

**Input:** `rgb (N, 3)`
**Output:** `bvp (N,)`

---

#### `omit.py` — OMIT (Orthogonal Matrix Image Transformation)

> Álvarez Casado, C., & Bordallo López, M. *Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.* IEEE JBHI (2023).

Uses QR decomposition to remove the dominant component of the RGB signal (noise/illumination) and extract the pulse in the orthogonal subspace.

```
A = rgb.T                    # (3, N)
Q, R = qr(A)
S = Q[:, 0]                  # dominant direction
P = I - Sᵀ·S                 # orthogonal projector
Y = P @ A                    # dominant component removed
BVP = Y[1, :]                # second row
```

**Input:** `rgb (N, 3)`
**Output:** `bvp (N,)`
**Note:** Robust to video compression artifacts (H.264).

---

#### `pos_wang.py` — POS (Plane-Orthogonal-to-Skin)

> Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. *Algorithmic principles of remote PPG.* IEEE TBME, 64(7), 1479–1491 (2017).

1.6 s sliding window. In each window, temporally normalizes the RGB and projects onto a plane orthogonal to the skin tone to separate the pulse from intensity variations.

```
l = ceil(1.6 × fs)               # frames per window
Cn = RGB[m:n] / mean(RGB[m:n])   # temporal normalization
S = [[0,1,-1],[-2,1,1]] @ Cn     # POS projection
h = S[0] + σ(S[0])/σ(S[1]) × S[1]  # alpha tuning
H[m:n] += h - mean(h)            # overlap-add
```

**Input:** `rgb (N, 3)`, `fs (float)`
**Output:** `bvp (N,)`
**Note:** `fs` required to compute the window length.

---

#### `adaptive_lms.py` — Adaptive LMS with IMU

> Widrow, B. & Hoff, M. E. *Adaptive switching circuits.* IRE WESCON (1960).

Cancels motion artifacts using the IMU signal as a noise reference. NLMS (Normalized LMS) filtering: the filter adapts its weights to estimate the motion component in the green signal and subtracts it.

```
green = rgb[:, 1]                        # green channel
imu_ref = interp(IMU, N)                 # interpolated to N frames
error[n] = green[n] - w·x[n]            # clean signal
w += (μ / (||x||² + ε)) × error × x    # NLMS weight update
```

**Input:** `rgb (N, 3)`, `imu_data (dict with ax/ay/az/gx/gy/gz)`
**Output:** `bvp (N,)`
**Parameters:** `μ=0.01` (step size), `ε=1e-6` (regularization)
**Note:** Drone mode only — requires IMU. Not available with webcam.

---

## Dependencies

```
opencv-python
mediapipe
requests
numpy
scipy
matplotlib
```

Install:
```bash
pip install opencv-python mediapipe requests numpy scipy matplotlib
```

> Requires **Python 3.11** (MediaPipe has limited compatibility with newer versions)

---

## How to Run

```bash
python main.py
```

The program prompts:
1. Video source: `A` (drone, `192.168.4.1`) or `B` (PC webcam)
2. rPPG algorithm: `GREEN`, `OMIT`, `POS_WANG` or `LMS` (LMS only available with drone)

Press `q` to stop — closes the video, runs the selected algorithm on the collected signal and displays the filtered BVP plot with the estimated BPM.

---

## Status and Next Steps

- [x] MJPEG stream from XIAO ESP32 camera
- [x] Face detection with MediaPipe FaceMesh
- [x] Forehead ROI (4 landmarks, pixel mask)
- [x] `rgb_signal (N, 3)` extraction frame by frame
- [x] Live RGB plot (sliding window of 300 frames)
- [x] FPS estimation from real timestamps
- [x] GREEN, OMIT, POS_WANG and LMS algorithms in `algoritmos/`
- [x] Filters in Processor: detrend + Butterworth bandpass [0.75–4.0 Hz]
- [x] Real-time HR (every 30 frames, minimum 30 s window)
- [x] `estimate_hr(bvp, fs)` → BPM via FFT (range 45–240 bpm)
- [x] LMS algorithm with IMU for motion artifact cancellation
- [ ] IMU-based motion compensation for GREEN / OMIT / POS algorithms
- [ ] Clinical validation of HR estimate vs. reference
