# Drone rPPG

Sistema de monitorização remota de sinais vitais (rPPG — *remote Photoplethysmography*) embarcado num drone, usando o XIAO ESP32-S3 Sense como unidade de aquisição e um PC como estação de processamento.

## Estrutura do Projeto

```
Drone_rPPG/
├── firmware/                # Código C (ESP-IDF) para o XIAO ESP32-S3 Sense
│   ├── main/
│   │   ├── main.c           # Wi-Fi AP, câmara OV3660, IMU, bateria, HTTP/WS
│   │   ├── fc.c             # Flight controller: Mahony AHRS + PID cascata + motor mixing
│   │   ├── fc.h             # API pública do flight controller
│   │   └── CMakeLists.txt
│   ├── CMakeLists.txt
│   ├── FIRMWARE.md          # Documentação técnica do firmware
│   ├── sdkconfig
│   └── partitions.csv
│
├── software/                # Código Python (PC / Estação Terrena)
│   ├── main.py              # Ponto de entrada: seleciona fonte e algoritmo
│   ├── DataHandler.py       # CameraHandler, WebcamHandler, IMUHandler
│   ├── Processor.py         # FaceProcessor: ROI, RGB, HR em tempo real
│   ├── SOFTWARE.md          # Documentação técnica do software
│   └── algoritmos/
│       ├── green.py         # Algoritmo GREEN (Verkruysse 2008)
│       ├── omit.py          # Algoritmo OMIT (Face2PPG, Casado 2023)
│       ├── pos_wang.py      # Algoritmo POS (Wang et al. 2017)
│       └── adaptive_lms.py  # Filtro adaptativo LMS com IMU (cancelamento de movimento)
│
└── README.md
```

## Hardware

| Componente | Modelo | Notas |
|------------|--------|-------|
| MCU | Seeed Studio XIAO ESP32-S3 Sense | 8 MB PSRAM + 8 MB Flash |
| Câmara | OV3660 | QQVGA 160×120, ~25 FPS, MJPEG |
| IMU | MPU-6050 | I2C, 500 Hz, ±16 g / ±2000°/s |
| Motores | 4× brushed DC | PWM LEDC 15 kHz, 8-bit (GPIO 2/3/4/7) |
| Bateria | LiPo 1S | Monitorização ADC em GPIO 1 |

## Firmware

Desenvolvido com **ESP-IDF v5.5.3**. O firmware corre no ESP32-S3 e é responsável por:

- Criar rede Wi-Fi AP (`rPPG-Drone`, `192.168.4.1`)
- Capturar e servir stream **MJPEG** da câmara OV3660 (porta 81)
- Expor dados IMU e atitude estimada via **HTTP `/imu`** (porta 80)
- Controlar o voo via **flight controller** em `fc.c` (Mahony AHRS + PID cascata, 500 Hz, Core 1)
- Monitorizar bateria com proteção por deep sleep

> **Importante:** AWB, AGC e AEC estão **desativados intencionalmente**. Controlos automáticos do sensor suprimem as variações subtis de cor da pele que constituem o sinal rPPG.

### Build e Flash

```bash
cd firmware
idf.py build flash monitor
```

### Controlo de Voo

A página web em `http://192.168.4.1` inclui joystick virtual (WebSocket a 20 Hz) e botões ARM/DISARM. Ver [FIRMWARE.md](firmware/FIRMWARE.md) para detalhes completos.

## Software (PC)

Desenvolvido em **Python 3.11** com OpenCV, MediaPipe e NumPy/SciPy.

| Módulo | Função |
|--------|--------|
| `main.py` | Seleciona fonte (drone / webcam) e algoritmo rPPG |
| `DataHandler.py` | `CameraHandler` (MJPEG), `WebcamHandler` (webcam), `IMUHandler` (polling `/imu`) |
| `Processor.py` | FaceMesh → ROI testa → RGB/frame → HR em tempo real |
| `algoritmos/green.py` | Canal verde direto |
| `algoritmos/omit.py` | Decomposição QR, subespaço ortogonal |
| `algoritmos/pos_wang.py` | Janela deslizante 1.6 s, projeção POS |
| `algoritmos/adaptive_lms.py` | NLMS adaptativo com IMU como referência de ruído (drone only) |

### Executar

```bash
cd software
pip install opencv-python mediapipe requests numpy scipy matplotlib
python main.py
```

O programa pergunta a fonte de vídeo (drone ou webcam) e o algoritmo rPPG a usar. Premir `q` para terminar — corre a análise final e mostra o plot BVP filtrado + BPM estimado.

## Princípio de Funcionamento

```
OV3660 (MJPEG)  ──→  CameraHandler  ──→  FaceProcessor
MPU-6050        ──→  IMUHandler     ──┘
                                         │
                                    FaceMesh → ROI testa
                                         │
                                    Média RGB / frame
                                         │
                              GREEN / OMIT / POS / LMS
                                         │
                              Detrend + Butterworth [0.75–4 Hz]
                                         │
                                    FFT → BPM
```

Ver [FIRMWARE.md](firmware/FIRMWARE.md) e [SOFTWARE.md](software/SOFTWARE.md) para documentação detalhada.
