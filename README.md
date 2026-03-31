# Drone rPPG

Drone-based remote vital sign monitoring system (rPPG — *remote Photoplethysmography*), using the XIAO ESP32-S3 Sense as the acquisition unit and a PC as the ground station.

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
│   ├── main.py              # Entry point: selects mode, source, algorithm, ROI and display
│   ├── DataHandler.py       # CameraHandler, WebcamHandler, IMUHandler
│   ├── Processor.py         # FaceProcessor: ROI, RGB extraction, real-time HR (rPPG)
│   ├── RespiratoryProcessor.py  # RespiratoryProcessor: pose tracking, real-time RR
│   ├── ROIExtraction.py     # ROI modes, face polygon, 9×9 grid, DMRS selection
│   ├── evaluate.py          # Offline evaluation: MAE, RMSE, PCC from CSV
│   ├── SOFTWARE.md          # Software technical documentation
│   ├── algoritmos/
│   │   ├── green.py         # GREEN algorithm (Verkruysse 2008)
│   │   ├── omit.py          # OMIT algorithm (Face2PPG, Casado 2023)
│   │   ├── pos_wang.py      # POS algorithm (Wang et al. 2017)
│   │   └── adaptive_lms.py  # Adaptive LMS filter with IMU (motion cancellation)
│   └── validation/
│       ├── SensorIntegration.ino        # Arduino sketch: PPG + respiration sensors → Serial
│       ├── PPG2csv.py                   # Record rPPG + hardware PPG simultaneously → CSV
│       ├── Resp2csv.py                  # Record RR + hardware respiration sensor → CSV
│       ├── Validation_rPPG_PPG.py       # Live rPPG vs hardware PPG comparison
│       ├── Validation_Resp_Resp.py      # Live RR vs hardware respiration comparison
│       └── Validation_rPPG_SMARTLOCK.py # rPPG vs SMARTLOCK reference device
│
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
| `main.py` | Selects mode (rPPG / Respiratory), source, algorithm, ROI and display mode at startup |
| `DataHandler.py` | `CameraHandler` (MJPEG), `WebcamHandler` (webcam), `IMUHandler` (polling `/imu`) |
| `Processor.py` | FaceMesh → ROI → RGB/frame → real-time HR (rPPG mode) |
| `RespiratoryProcessor.py` | Bartula 2013: chest ROI → 1D profile cross-correlation → real-time RR |
| `ROIExtraction.py` | ROI modes, polygon extraction, 9×9 grid, DMRS region selection |
| `evaluate.py` | MAE / RMSE / PCC evaluation from a ground-truth CSV |
| `algoritmos/green.py` | Direct green channel |
| `algoritmos/omit.py` | QR decomposition, orthogonal subspace |
| `algoritmos/pos_wang.py` | 1.6 s sliding window, POS projection |
| `algoritmos/adaptive_lms.py` | Adaptive NLMS with IMU as noise reference (drone only) |
| `validation/PPG2csv.py` | Simultaneous rPPG + hardware PPG recording → CSV |
| `validation/Resp2csv.py` | Simultaneous RR + hardware respiration sensor recording → CSV |
| `validation/Validation_rPPG_PPG.py` | Live comparison of rPPG vs hardware PPG reference |
| `validation/Validation_Resp_Resp.py` | Live comparison of RR vs hardware respiration reference |
| `validation/Validation_rPPG_SMARTLOCK.py` | rPPG vs SMARTLOCK reference device |
| `validation/SensorIntegration.ino` | Arduino sketch: reads PPG + respiration sensors, streams over Serial at 115200 baud |

### ROI Modes

| Mode | Description |
|------|-------------|
| **Forehead** | 4-landmark forehead quadrilateral (original, fastest) |
| **Full face** | 36-landmark face oval contour |
| **Multi-region** | Face oval divided into 9×9 grid; DMRS dynamically selects the best regions (Face2PPG) |

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
4. *(rPPG only)* Algorithm and ROI

Press `q` to stop.

| Mode | Measures | Method |
|------|----------|--------|
| **rPPG** | Heart rate (BPM) | FaceMesh → face ROI → colour signal |
| **Respiratory** | Breathing rate (rpm) | Bartula 2013 — chest ROI cross-correlation, no landmarks |

### Evaluate

```bash
python evaluate.py results.csv --plot --save
```

Expects a CSV with columns `HR_gt`, `HR_GREEN`, `HR_OMIT`, `HR_POS_WANG`, `HR_LMS`, `roi_mode`, `source`. Outputs MAE / RMSE / PCC tables per algorithm and condition.

## How It Works

**rPPG mode (heart rate):**
```
OV3660 (MJPEG)  ──→  CameraHandler  ──→  FaceProcessor
MPU-6050        ──→  IMUHandler     ──┘
                                         │
                                    FaceMesh (468 landmarks)
                                         │
                          ┌──────────────┼──────────────┐
                       Forehead       Face oval     Multi-region
                      (4 landmarks) (36 landmarks)  (9×9 DMRS)
                          └──────────────┼──────────────┘
                                         │
                                   Mean RGB / frame
                                         │
                              GREEN / OMIT / POS / LMS
                                         │
                              Detrend + Butterworth [0.75–4 Hz]
                                         │
                                    FFT → BPM
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
                           Peak detection → breath-by-breath validation
                           (exclude motion segments, invalid durations)
                                         │
                           Median inter-breath interval → rpm
```

See [FIRMWARE.md](firmware/FIRMWARE.md) and [SOFTWARE.md](software/SOFTWARE.md) for detailed documentation.
