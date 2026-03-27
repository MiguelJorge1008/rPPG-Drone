# Firmware — Drone rPPG (XIAO ESP32-S3 Sense)

Firmware for MJPEG video capture over Wi-Fi with the OV3660 sensor and brushed motor flight control, optimized for **rPPG (Remote Photoplethysmography)** on a drone.

---

## Hardware

| Component | Model |
|-----------|-------|
| MCU | Seeed Studio XIAO ESP32-S3 Sense |
| Camera | OV3660 (QQVGA 160×120, ~25 FPS, MJPEG) |
| IMU | MPU-6050 (I2C, 500 Hz, ±16 g / ±2000°/s) |
| Motors | 4× brushed DC (LEDC PWM 15 kHz, 8-bit) |
| Battery | LiPo 1S (ADC monitoring GPIO 1) |
| PSRAM | 8 MB (Octal) |
| Flash | 8 MB |
| Framework | ESP-IDF v5.5.3 |

---

## Network

The firmware operates in **Access Point (AP) mode** — the ESP32 creates its own Wi-Fi network.

| Parameter | Value |
|-----------|-------|
| SSID | `rPPG-Drone` |
| Password | `12345678` |
| Fixed IP | `192.168.4.1` |
| Channel | 2 (2.4 GHz) |
| Security | WPA2-PSK |

---

## Quick Start

1. Build and flash:
```bash
idf.py build flash monitor
```

2. Connect your PC/phone to the `rPPG-Drone` network (password: `12345678`)

3. Open a browser at `http://192.168.4.1`

---

## HTTP Endpoints

There are **two HTTP servers**:
- **Port 80** — HTML page, controls, IMU, flight
- **Port 81** — MJPEG stream (dedicated server, does not block other endpoints)

| URL | Method | Server | Description |
|-----|--------|--------|-------------|
| `/` | GET | :80 | HTML page with stream + camera, flight and PID controls |
| `/stream` | GET | :81 | Continuous MJPEG stream |
| `/control?aec=X&gain=Y` | GET | :80 | Adjust exposure and gain at runtime |
| `/imu` | GET | :80 | JSON with raw IMU, estimated attitude, battery and FC state |
| `/cmd?state=arm\|disarm` | GET | :80 | Arm/disarm the flight controller |
| `/cmd?roll=&pitch=&yaw=&thrust=` | GET | :80 | Manual flight setpoint |
| `/pid[?att_kp=&...]` | GET | :80 | Read or update PID gains at runtime |
| `/ws` | WS | :80 | WebSocket for joystick setpoints at 20 Hz |

### `/control` Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `aec` | 0 – 1200 | Manual exposure time (OV3660) |
| `gain` | 0 – 30 | Manual analog gain |

### `/imu` Response

```json
{
  "ax": 0.01, "ay": -0.02, "az": 0.99,
  "gx": 10.5, "gy": -5.2, "gz": 0.1,
  "roll": 1.2, "pitch": -0.8, "yaw": 45.0,
  "bat": 3850,
  "fc_state": "disarmed"
}
```

| Field | Unit | Description |
|-------|------|-------------|
| `ax`, `ay`, `az` | g | Linear acceleration (±16 g, 2048 LSB/g) |
| `gx`, `gy`, `gz` | deg/s | Angular velocity (±2000°/s, 16.4 LSB/°/s) |
| `roll`, `pitch`, `yaw` | degrees | Attitude estimated by Mahony AHRS |
| `bat` | mV | Battery voltage (100K+100K divider, GPIO1) |
| `fc_state` | string | `"disarmed"` or `"flight"` |

### `/cmd` Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `state` | `arm` / `disarm` | FC state transition |
| `roll` | −30 – +30 | Desired roll angle (degrees) |
| `pitch` | −30 – +30 | Desired pitch angle (degrees) |
| `yaw` | −200 – +200 | Desired yaw rate (°/s) |
| `thrust` | 0.0 – 1.0 | Normalized base thrust |

### `/pid` Parameters

GET without query → returns JSON with current gains.
GET with query → applies new gains and returns confirmed values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `att_kp` | Outer loop kp (roll = pitch) | 3.5 |
| `att_ki` | Outer loop ki | 2.0 |
| `rr_kp/ki/kd` | Rate roll kp/ki/kd | 70 / 0 / 0 |
| `rp_kp/ki/kd` | Rate pitch kp/ki/kd | 70 / 0 / 0 |
| `ry_kp/ki/kd` | Rate yaw kp/ki/kd | 50 / 0 / 0 |

### WebSocket `/ws`

The client sends JSON at 20 Hz:
```json
{"r": 5.0, "p": -3.0, "y": 10.0, "t": 0.55}
```

| Field | Unit | Description |
|-------|------|-------------|
| `r` | degrees | Roll (−30 to +30) |
| `p` | degrees | Pitch (−30 to +30) |
| `y` | °/s | Yaw rate (−200 to +200) |
| `t` | 0.0 – 1.0 | Thrust |

Each WS frame resets the FC watchdog. If no frames arrive for **500 ms**, the FC automatically transitions to `DISARMED`.

---

## Flight Controller

The flight controller is implemented in `main/fc.c` + `main/fc.h` and runs in a dedicated FreeRTOS task on **Core 1** at **500 Hz** (priority 8).

### Architecture

```
MPU-6050 (500 Hz)
    │
    ▼
Mahony AHRS  →  Euler angles (roll, pitch, yaw)
    │
    ▼
Outer PID (attitude)  →  rate setpoint
    │
    ▼
Inner PID (rate)  →  torques (roll, pitch, yaw)
    │
    ▼
Motor mixing  →  M1/M2/M3/M4 duties (0–255)
```

**Sensor fusion:** Mahony AHRS (ported from `sensfusion6.c` — Crazyflie / ESP-Drone).
**PID:** Cascade rate/attitude (ported from `pid.c` — Crazyflie / ESP-Drone).

### FC States

| State | Description |
|-------|-------------|
| `FC_DISARMED` | Motors off, PIDs reset |
| `FC_FLIGHT` | PID active, accepts setpoints via `/cmd` or `/ws` |

### Safety

| Mechanism | Value | Behavior |
|-----------|-------|----------|
| Setpoint watchdog | 500 ms | FC transitions to DISARMED if no setpoint received |
| Tilt protection | ±50° | FC transitions to DISARMED if roll or pitch exceeds limit |
| Mahony warmup | 2 s | Motors locked at boot for AHRS convergence |
| Critical battery | < 3100 mV | FC transitions to DISARMED + deep sleep |

### Motor Mixing (quadrotor X)

```
M1 (front-left, CW)  |  M2 (front-right, CCW)
─────────────────────────────────────────────
M4 (back-left, CCW)  |  M3 (back-right, CW)
```

| Motor | GPIO | Rotation | LEDC Channel |
|-------|------|----------|--------------|
| M1 (front-left) | GPIO 2 | CW | CH1 |
| M2 (front-right) | GPIO 3 | CCW | CH2 |
| M3 (back-right) | GPIO 4 | CW | CH3 |
| M4 (back-left) | GPIO 7 | CCW | CH4 |

**LEDC:** Timer 1, 15 kHz, 8-bit (0–255). Timer 0 / Channel 0 is reserved for the camera XCLK.

---

## Battery

| Parameter | Value |
|-----------|-------|
| ADC Pin | GPIO 1 (ADC1 CH0) |
| Voltage divider | 100 kΩ + 100 kΩ (factor ×2) |
| Warning | < 3500 mV |
| Critical / deep sleep | < 3100 mV |
| No battery | < 2000 mV (alarm ignored) |
| Status LED | GPIO 21 |

When battery enters critical: FC transitions to DISARMED → LED blinks 10× → `esp_deep_sleep_start()`.

---

## OV3660 Sensor Configuration for rPPG

### Why Disable Automatic Controls

For rPPG it is **mandatory** to disable all automatic sensor controls. If active, the sensor continuously compensates for brightness variations in the skin — which is precisely the signal being captured.

| Control | Disabled function | Reason |
|---------|-------------------|--------|
| AWB (Auto White Balance) | `set_whitebal(s, 0)` | Would re-balance R/G/B channels frame by frame |
| AGC (Auto Gain Control) | `set_gain_ctrl(s, 0)` | Gain variations introduce artifacts in the PPG signal |
| AEC (Auto Exposure) | `set_exposure_ctrl(s, 0)` | AEC compensates exactly the skin reflectance variations |
| AEC nightmode | `set_aec2(s, 0)` | Extra DSP mode of AEC, equally harmful |

### Calibrated Default Values

```c
s->set_agc_gain(s, 3);        // Fixed gain
s->set_aec_value(s, 1200);    // Fixed exposure — adjustable at runtime via /control
```

| Condition | Recommended aec | gain |
|-----------|-----------------|------|
| Outdoors — direct sunlight | 20 – 60 | 0 – 2 |
| Outdoors — overcast | 100 – 200 | 2 – 4 |
| Indoors (testing) | 400 – 800 | 4 – 8 |

### Green Cast — Expected Behavior

With AWB disabled, the image shows a **pronounced green cast**. This is normal:
- The OV3660 sensor uses **RGGB** Bayer pattern — 2 green pixels per 1 red and 1 blue
- Without automatic color correction, green dominates (raw sensor response)
- The green channel is the **most sensitive to the PPG signal** (~550 nm)
- Most rPPG algorithms (GREEN, CHROM, POS) extract the signal primarily from the green channel

### Indoor Lighting Flicker

In indoor environments with LED/fluorescent lighting, periodic brightness variations may appear. This is due to the mains frequency (100 Hz in Europe / 50 Hz AC).

- If variation occurs in non-skin areas → it is lighting flicker
- If variation occurs mainly in skin periodically (~60–80 bpm) → it is the rPPG signal

In outdoor sunlight this problem does not exist.

---

## Performance

| Parameter | Value |
|-----------|-------|
| Resolution | **QQVGA (160×120)** |
| Average FPS | **~25 FPS** (dependent on Wi-Fi conditions) |
| JPEG quality | 8 |
| Frame buffers | 3 in PSRAM (`CAMERA_GRAB_LATEST`) |
| XCLK | 24 MHz |
| Serial log | `>>> STREAMING: XX.XX FPS \| Res: QQVGA` (every second) |

---

## HTTP Server

```c
/* Port 80 — page + controls */
server_config.stack_size       = 8192;
server_config.max_uri_handlers = 10;
server_config.max_open_sockets = 7;

/* Port 81 — MJPEG stream */
stream_config.stack_size       = 8192;
stream_config.max_uri_handlers = 2;
stream_config.max_open_sockets = 2;
```

The MJPEG stream runs on a separate server (port 81) so that requests to `/control`, `/imu`, `/cmd`, etc. are never blocked by the persistent TCP socket of the stream.

The HTML page pauses the stream before sending the `/control` request and reconnects 200 ms later — prevents image freezing due to socket contention.

---

## Pin Mapping (XIAO ESP32-S3 Sense)

### Camera (OV3660)

| Signal | GPIO |
|--------|------|
| XCLK | 10 |
| SDA (SCCB) | 40 |
| SCL (SCCB) | 39 |
| D0–D7 | 15, 17, 18, 16, 14, 12, 11, 48 |
| VSYNC | 38 |
| HREF | 47 |
| PCLK | 13 |
| PWDN / RESET | N/A (-1) |

### IMU (MPU-6050)

| Signal | GPIO |
|--------|------|
| SCL (I2C) | 5 |
| SDA (I2C) | 6 |
| I2C Address | 0x68 |
| Sample rate | 500 Hz |
| Gyro full-scale | ±2000°/s (16.4 LSB/°/s) |
| Accel full-scale | ±16 g (2048 LSB/g) |

### Motors (LEDC PWM)

| Signal | GPIO |
|--------|------|
| M1 (front-left, CW) | 2 |
| M2 (front-right, CCW) | 3 |
| M3 (back-right, CW) | 4 |
| M4 (back-left, CCW) | 7 |

### Other

| Signal | GPIO |
|--------|------|
| Battery ADC | 1 (ADC1 CH0) |
| Status LED | 21 |

---

## Troubleshooting

**Browser only shows the stream without controls**
→ Old version cached. Press Ctrl+Shift+R or open in incognito mode.

**Stream not appearing (`http://192.168.4.1`)**
→ The stream is on port 81. The HTML page points to `http://192.168.4.1:81/stream`. Check if the second server started in the serial logs.

**Compile time in serial monitor is old**
→ Firmware was not reflashed. Run `idf.py build flash monitor`.

**Camera controls do not change the image**
→ The JS pauses the stream, sends `/control`, reconnects. Check serial logs: `Control: OK | aec=X gain=Y`.

**Forehead overexposed / saturated**
→ Reduce `aec` via slider. For outdoor sunlight, use aec < 60.

**`rPPG-Drone` network not appearing**
→ Check serial monitor: `AP started: rPPG-Drone | 192.168.4.1`. If not shown, reflash.

**FC does not arm (ARM button has no effect)**
→ Wait 2 s after boot (Mahony warmup). Check that MPU-6050 initialized without error in serial logs.

**FC disarms during flight**
→ May be watchdog (no setpoints for more than 500 ms) or tilt protection (tilt > 50°). Check serial logs: `WDT timeout` or `TILT PROTECTION`.

**No internet access while connected to ESP32**
→ Expected behavior — the ESP32 AP does not provide internet. Prepare everything before switching networks.
