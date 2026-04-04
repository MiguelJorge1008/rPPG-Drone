# Drone rPPG

Drone-based remote vital sign monitoring system (rPPG — *remote Photoplethysmography*), using the XIAO ESP32-S3 Sense as the acquisition unit and a PC as the ground station.

<p align="center">
  <img src="assets/drone.jpg" alt="Drone prototype" width="480"/>
</p>

## Project Structure

```
Drone_rPPG/
├── firmware/                # C code (ESP-IDF) for the XIAO ESP32-S3 Sense
│   ├── main/
│   │   ├── main.c           # Wi-Fi AP, OV3660 camera, IMU, battery, HTTP/WS
│   │   ├── fc.c             # Flight controller: Mahony AHRS + cascade PID + motor mixing
│   │   ├── fc.h             # Flight controller public API
│   │   └── CMakeLists.txt
│   ├── CMakeLists.txt
│   ├── FIRMWARE.md          # Firmware technical documentation
│   ├── sdkconfig
│   └── partitions.csv
│
├── software/                # Python code (PC / Ground Station)
│   ├── main.py              # Entry point: selects mode, source, algorithm and display
│   ├── DataHandler.py       # CameraHandler, WebcamHandler, IMUHandler
│   ├── Processor.py         # FaceProcessor: forehead ROI, RGB extraction, real-time HR (rPPG)
│   ├── RespiratoryProcessor.py  # RespiratoryProcessor: pose tracking, real-time RR
│   ├── ROIExtraction.py     # Forehead ROI polygon with EMA smoothing
│   ├── SOFTWARE.md          # Software technical documentation
│   ├── algoritmos/
│   │   ├── green.py         # GREEN algorithm (Verkruysse 2008)
│   │   ├── omit.py          # OMIT algorithm (Face2PPG, Casado 2023)
│   │   ├── pos_wang.py      # POS algorithm (Wang et al. 2017)
│   │   └── adaptive_lms.py  # Adaptive LMS filter with IMU (motion cancellation)
│   └── test/
│       ├── RecordPPG.py     # Record rPPG + hardware PPG simultaneously → CSV
│       ├── RecordResp.py    # Record RR + hardware respiration sensor → CSV
│       ├── ComputeMetrics.py  # Compute HR/HRV/RR metrics → data_processed/
│       ├── Evaluate.py      # Offline evaluation: MAE±SD per algorithm
│       ├── RunTest.py       # Full pipeline: ComputeMetrics → Evaluate
│       ├── Arduino/
│       │   └── SensorIntegration.ino  # Arduino: PPG + resp sensors → Serial 115200
│       ├── data/            # Raw CSV recordings (signal only)
│       ├── data_processed/  # Processed CSVs with computed metrics
│       └── results/         # Evaluation plots (PNG)
│
├── assets/                  # Project images
└── README.md
```

## Hardware

| Component | Model | Notes |
|-----------|-------|-------|
| MCU | Seeed Studio XIAO ESP32-S3 Sense | 8 MB PSRAM + 8 MB Flash |
| Camera | OV3660 | QVGA 320×240, ~25 FPS, MJPEG |
| IMU | MPU-6050 | I2C, 500 Hz, ±16 g / ±2000°/s |
| Motors | 4× brushed DC | LEDC PWM 15 kHz, 8-bit (GPIO 2/3/4/7) |
| Battery | LiPo 1S | ADC monitoring on GPIO 1 |

## Firmware

Developed with **ESP-IDF v5.5.3**. The firmware runs on the ESP32-S3 and is responsible for:

- Creating a Wi-Fi AP network (`rPPG-Drone`, `192.168.4.1`)
- Capturing and serving the OV3660 camera **MJPEG stream** (port 81)
- Exposing IMU data and estimated attitude via **HTTP `/imu`** (port 80)
- Flight control via **flight controller** in `fc.c` (Mahony AHRS + cascade PID, 500 Hz, Core 1)
- Battery monitoring with deep sleep protection

> **Important:** AWB, AGC and AEC are **intentionally disabled**. Automatic sensor controls suppress the subtle skin color variations that make up the rPPG signal.

### Build and Flash

```bash
cd firmware
idf.py build flash monitor
```

### Flight Control

The web page at `http://192.168.4.1` includes a virtual joystick (WebSocket at 20 Hz) and ARM/DISARM buttons. See [FIRMWARE.md](firmware/FIRMWARE.md) for full details.

## Software (PC)

Developed in **Python 3.11** with OpenCV, MediaPipe and NumPy/SciPy.

| Module | Function |
|--------|----------|
| `main.py` | Selects mode (rPPG / Respiratory), source, algorithm and display mode at startup |
| `DataHandler.py` | `CameraHandler` (MJPEG), `WebcamHandler` (webcam), `IMUHandler` (polling `/imu`) |
| `Processor.py` | FaceMesh → forehead ROI → RGB/frame → real-time HR via peak detection (30 s window) |
| `RespiratoryProcessor.py` | Bartula 2013: chest ROI → 1D profile cross-correlation → real-time RR |
| `ROIExtraction.py` | Forehead polygon (4 landmarks) with EMA smoothing |
| `algoritmos/green.py` | Direct green channel |
| `algoritmos/omit.py` | QR decomposition, orthogonal subspace |
| `algoritmos/pos_wang.py` | 1.6 s sliding window, POS projection |
| `algoritmos/adaptive_lms.py` | Adaptive NLMS with IMU as noise reference (drone only) |
| `test/RecordPPG.py` | Simultaneous rPPG + hardware PPG recording → CSV |
| `test/RecordResp.py` | Simultaneous RR + hardware respiration sensor recording → CSV |
| `test/ComputeMetrics.py` | Post-process raw CSVs → metrics CSVs in `data_processed/` |
| `test/Evaluate.py` | MAE±SD per algorithm from `data_processed/`; plots to `results/` |
| `test/RunTest.py` | Full pipeline: ComputeMetrics → Evaluate |
| `test/Arduino/SensorIntegration.ino` | Arduino: reads PPG + respiration sensors, streams over Serial at 115200 baud |

### Run

```bash
cd software
pip install opencv-python mediapipe requests numpy scipy matplotlib pandas
python main.py
```

The program prompts for:
1. Mode: `1` rPPG · `2` Respiratory
2. Display: `1` graphs + camera window · `2` terminal only (higher FPS)
3. Video source: `A` drone · `B` webcam
4. *(rPPG only)* Algorithm: GREEN, OMIT, POS_WANG, LMS

Press `q` to stop.

| Mode | Measures | Method |
|------|----------|--------|
| **rPPG** | Heart rate (BPM) | FaceMesh → forehead ROI → colour signal → peak detection |
| **Respiratory** | Breathing rate (RPM) | Bartula 2013 — chest ROI cross-correlation, no landmarks |

### Record and Evaluate

<p align="center">
  <img src="assets/sensor_board.jpg" alt="Arduino sensor board (PPG + respiration)" width="480"/>
</p>

```bash
# From software/test/
python RecordPPG.py    # rPPG + hardware PPG  → data/BVP_<ALGO>_<ts>_{sw,hw}.csv
python RecordResp.py   # RR  + hardware resp  → data/RESP_<ts>_{sw,hw}.csv

# Run full evaluation pipeline
python RunTest.py
```

`RunTest.py` runs in two steps:

1. **`ComputeMetrics.py`** — reads raw signals from `data/`, computes HR/HRV/RR metrics via peak detection, writes processed CSVs to `data_processed/`
2. **`Evaluate.py`** — reads `data_processed/`, interpolates hardware GT onto software timestamps, prints per-algorithm MAE±SD tables and saves plots to `results/`

```
Table 1 — HR evaluation (MAE ± SD)
  GREEN · OMIT · POS_WANG

Table 2 — Respiratory rate evaluation (MAE ± SD)
  BARTULA

Table 3 — HRV evaluation (SDNN / RMSSD MAE ± SD)
  GREEN · OMIT · POS_WANG
```

## How It Works

**rPPG mode (heart rate):**
```
OV3660 (MJPEG)  ──→  CameraHandler  ──→  FaceProcessor
MPU-6050        ──→  IMUHandler     ──┘
                                         │
                                    FaceMesh (468 landmarks)
                                         │
                                    Forehead ROI (4 landmarks)
                                         │
                                   Mean RGB / frame
                                         │
                              GREEN / OMIT / POS / LMS
                                         │
                              Detrend + Butterworth [0.75–4 Hz]
                                         │
                               find_peaks → BPM (30 s window)
```

**Respiratory mode (breathing rate) — Bartula et al. 2013:**
```
OV3660 (MJPEG)  ──→  CameraHandler  ──→  RespiratoryProcessor
                                         │
                                 ┌── STARTUP (once) ──┐
                                 │  MediaPipe Pose     │
                                 │  detect shoulders   │
                                 │  → chest ROI (px)  │
                                 │  close Pose         │
                                 └────────────────────┘
                                         │
                                   Fixed chest ROI
                                         │
                           1D vertical profile (mean + std / row)
                           High-pass (removes illumination drift)
                                         │
                           Phase cross-correlation (Hann window)
                           r = F⁻¹( F(pₜ) · conj(F(pₜ₋₁)) )
                           Sub-pixel peak → frame shift
                                         │
                           Integrate shifts → chest position signal
                                         │
                           Global motion detector (block-based)
                                         │
                           Detrend + Butterworth [0.1–0.5 Hz]
                                         │
                           find_peaks → breath-by-breath validation
                           (exclude motion segments, invalid durations)
                                         │
                           Median inter-breath interval → RPM
```

See [FIRMWARE.md](firmware/FIRMWARE.md) and [SOFTWARE.md](software/SOFTWARE.md) for detailed documentation.
