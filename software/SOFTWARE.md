# Drone rPPG — Software

Software de aquisição de vídeo e deteção facial para extração de sinais rPPG (remote Photoplethysmography) a partir de um drone equipado com câmara XIAO ESP32.

---

## Estrutura de ficheiros

```
software/
├── main.py                  # Ponto de entrada: seleciona fonte e algoritmo
├── DataHandler.py           # CameraHandler, WebcamHandler, IMUHandler
├── Processor.py             # Deteção facial, ROI, filtros, HR em tempo real
├── algoritmos/
│   ├── green.py             # Algoritmo GREEN (Verkruysse 2008)
│   ├── omit.py              # Algoritmo OMIT (Casado & López 2023)
│   ├── pos_wang.py          # Algoritmo POS (Wang et al. 2017)
│   └── adaptive_lms.py      # Filtro NLMS adaptativo com IMU (drone only)
└── SOFTWARE.md
```

---

## Pipeline rPPG

```
Frame (câmara / webcam)
  → MediaPipe FaceMesh (deteção facial)
  → ROI testa (4 landmarks) → média RGB por frame
  → rgb_signal  (N, 3)
  → [GREEN / OMIT / POS_WANG / LMS]  →  BVP bruto  (N,)
  → apply_filters  →  BVP filtrado  (N,)
  → estimate_hr  →  BPM  (FFT, range 45–240 bpm)
```

---

## Ficheiros

### `main.py`

Ponto de entrada. Pergunta ao utilizador a fonte de vídeo e o algoritmo rPPG, instancia os handlers e arranca o loop principal.

- **Fonte A — Drone:** `CameraHandler` (stream MJPEG) + `IMUHandler` (polling `/imu`); IP fixo `http://192.168.4.1`
- **Fonte B — Webcam:** `WebcamHandler` (OpenCV VideoCapture local)
- **Algoritmos disponíveis:** GREEN, OMIT, POS_WANG, LMS (LMS só disponível no modo drone, requer IMU)

---

### `DataHandler.py`

Três classes de aquisição de dados.

**Classe: `CameraHandler`** — stream MJPEG da câmara do drone

| Método | Descrição |
|---|---|
| `__init__(base_url)` | Inicia a ligação e arranca a thread de captura |
| `update()` | Thread contínua que lê o stream MJPEG, extrai frames JPEG e guarda o último frame |
| `get_frame()` | Devolve o último frame capturado (`None` se ainda não houver) |
| `stop()` | Para a thread de captura |

**Detalhes técnicos:**
- Stream lido via `requests` com `stream=True`
- Frames JPEG extraídos do buffer binário pelos marcadores `0xFF 0xD8` (início) e `0xFF 0xD9` (fim)
- Thread separada (daemon) com reconexão automática (retry 2 s)
- URL do stream: `http://<IP>:81/stream`

**Classe: `WebcamHandler`** — webcam local (PC)

| Método | Descrição |
|---|---|
| `__init__()` | Abre `cv2.VideoCapture(0)` e arranca thread de captura |
| `get_frame()` | Devolve o último frame capturado |
| `stop()` | Liberta o VideoCapture |

**Classe: `IMUHandler`** — dados IMU do drone via HTTP

| Método | Descrição |
|---|---|
| `__init__(base_url)` | Arranca thread de polling ao endpoint `/imu` a 10 Hz |
| `get_imu()` | Devolve o último sample JSON ou `None` |
| `stop()` | Para a thread |

Campos do JSON: `ax, ay, az` (g), `gx, gy, gz` (°/s), `bat` (mV), `fc_state`.

---

### `Processor.py`

Processa os frames com MediaPipe FaceMesh, extrai o sinal RGB da testa, corre os algoritmos rPPG e filtra os sinais BVP.

**Classe: `FaceProcessor`**

| Método | Descrição |
|---|---|
| `__init__(camera, algorithm, imu)` | Inicializa FaceMesh, `rgb_signal`, timestamps e gráfico RGB em tempo real |
| `process_frame(frame)` | Redimensiona frame, corre FaceMesh, extrai RGB da ROI, desenha landmarks |
| `get_forehead_polygon(landmarks, w, h)` | Converte 4 landmarks da testa em coordenadas de pixel |
| `get_fps(window=60)` | Estima fps a partir dos timestamps reais dos últimos `window` frames; fallback 27.0 Hz |
| `apply_filters(bvp, fs)` | Detrend + Butterworth bandpass [0.75–4.0 Hz] sobre o sinal BVP |
| `_update_plot()` | Atualiza o gráfico RGB ao vivo (janela deslizante 300 frames) |
| `run()` | Loop principal: captura frames, processa, mostra vídeo e atualiza gráfico a cada 5 frames |
| `stop()` | Fecha FaceMesh, para câmara, corre o algoritmo selecionado e mostra plot BVP + BPM |

**Configuração do FaceMesh:**
- `max_num_faces=1` — deteta apenas uma face
- `refine_landmarks=False` — sem refinamento de íris
- `min_detection_confidence=0.5`
- `min_tracking_confidence=0.5`

**Otimizações de performance:**
- Frame redimensionado para **320×240** antes do MediaPipe
- `rgb.flags.writeable = False` antes do `process()` (otimização MediaPipe)
- Apenas pontos desenhados, sem ligações (`connections=None`)

**ROI da testa:**

Quadrilátero central definido por 4 landmarks: `[103, 332, 296, 66]`

| Índice | Posição |
|---|---|
| 103 | Topo esquerda |
| 332 | Topo direita |
| 296 | Fundo direita (acima sobrancelha) |
| 66 | Fundo esquerda (acima sobrancelha) |

Contorno amarelo com preenchimento semitransparente (20% opacidade).

**Sinal RGB (`rgb_signal`):**
- Por frame com face detetada: todos os píxeis dentro do polígono ROI → média espacial → `[R, G, B]`
- Acumulado em `self.rgb_signal`; convertido para `np.ndarray (N, 3)` no `stop()`

**Estimação de fps (`get_fps`):**
- Calculado a partir dos timestamps reais dos frames (`time.time()`)
- Evita depender do fps nominal da câmara, que pode variar no stream MJPEG
- Fallback: `27.0 Hz`

**Filtros (`apply_filters`):**

Aplicados no `Processor` após cada algoritmo, não dentro dos algoritmos. Conforme Face2PPG (Casado & López, 2023) e Wang et al. (2017), que descreve o POS core sem filtragem.

1. **Detrend** (λ=100) — remove deriva lenta da baseline
2. **Butterworth bandpass** [0.75–4.0 Hz], ordem 2, `filtfilt` — banda cardíaca (45–240 bpm)

**Output no `stop()`:**

Figura matplotlib com o BVP filtrado do algoritmo selecionado e o BPM estimado no título. O `fs` real (medido em runtime) é indicado.

---

### `algoritmos/`

Cada algoritmo recebe `rgb (N, 3)` (e opcionalmente `fs` ou `imu`) e devolve `bvp (N,)` — sinal BVP bruto, sem filtros.

#### `green.py` — GREEN

> Verkruysse, W., Svaasand, L. O. & Nelson, J. S. *Remote plethysmographic imaging using ambient light.* Optical Express 16, 21434–21445 (2008).

Extrai o canal verde diretamente. A hemoglobina absorve fortemente na banda verde, tornando-o o canal com maior amplitude pulsátil.

```
BVP = G
```

**Input:** `rgb (N, 3)`
**Output:** `bvp (N,)`

---

#### `omit.py` — OMIT (Orthogonal Matrix Image Transformation)

> Álvarez Casado, C., & Bordallo López, M. *Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.* IEEE JBHI (2023).

Utiliza decomposição QR para remover a componente dominante do sinal RGB (ruído/iluminação) e extrair o pulso no subespaço ortogonal.

```
A = rgb.T                    # (3, N)
Q, R = qr(A)
S = Q[:, 0]                  # direcção dominante
P = I - Sᵀ·S                 # projector ortogonal
Y = P @ A                    # componente dominante removida
BVP = Y[1, :]                # segunda linha
```

**Input:** `rgb (N, 3)`
**Output:** `bvp (N,)`
**Nota:** Robusto a artefactos de compressão de vídeo (H.264).

---

#### `pos_wang.py` — POS (Plane-Orthogonal-to-Skin)

> Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. *Algorithmic principles of remote PPG.* IEEE TBME, 64(7), 1479–1491 (2017).

Janela deslizante de 1.6 s. Em cada janela, normaliza temporalmente o RGB e projecta num plano ortogonal ao tom de pele para separar o pulso das variações de intensidade.

```
l = ceil(1.6 × fs)           # frames por janela
Cn = RGB[m:n] / mean(RGB[m:n])   # normalização temporal
S = [[0,1,-1],[-2,1,1]] @ Cn     # projecção POS
h = S[0] + σ(S[0])/σ(S[1]) × S[1]  # alpha tuning
H[m:n] += h - mean(h)        # overlap-add
```

**Input:** `rgb (N, 3)`, `fs (float)`
**Output:** `bvp (N,)`
**Nota:** `fs` necessário para calcular o comprimento da janela.

---

#### `adaptive_lms.py` — LMS adaptativo com IMU

> Widrow, B. & Hoff, M. E. *Adaptive switching circuits.* IRE WESCON (1960).

Cancela artefactos de movimento usando o sinal IMU como referência de ruído. Filtragem NLMS (Normalized LMS): o filtro adapta os seus pesos para estimar a componente de movimento no sinal verde e subtrai-a.

```
green = rgb[:, 1]            # canal verde
imu_ref = interp(IMU, N)     # interpolado para N frames
error[n] = green[n] - w·x[n]        # sinal limpo
w += (μ / (||x||² + ε)) × error × x # actualização NLMS
```

**Input:** `rgb (N, 3)`, `imu_data (dict com ax/ay/az/gx/gy/gz)`
**Output:** `bvp (N,)`
**Parâmetros:** `μ=0.01` (step size), `ε=1e-6` (regularização)
**Nota:** Exclusivo do modo drone — requer IMU. Não disponível com webcam.

---

## Dependências

```
opencv-python
mediapipe
requests
numpy
scipy
matplotlib
```

Instalar:
```bash
pip install opencv-python mediapipe requests numpy scipy matplotlib
```

> Requer **Python 3.11** (MediaPipe tem compatibilidade limitada com versões mais recentes)

---

## Como correr

```bash
python main.py
```

O programa pergunta:
1. Fonte de vídeo: `A` (drone, `192.168.4.1`) ou `B` (webcam PC)
2. Algoritmo rPPG: `GREEN`, `OMIT`, `POS_WANG` ou `LMS` (LMS só disponível com drone)

Premir `q` para terminar — fecha o vídeo, corre o algoritmo selecionado sobre o sinal recolhido e mostra o plot BVP filtrado com o BPM estimado.

---

## Estado e próximos passos

- [x] Stream MJPEG da câmara XIAO ESP32
- [x] Deteção facial com MediaPipe FaceMesh
- [x] ROI testa (4 landmarks, máscara de píxeis)
- [x] Extração de `rgb_signal (N, 3)` frame a frame
- [x] Gráfico RGB em tempo real (janela deslizante 300 frames)
- [x] Estimação de fps a partir de timestamps reais
- [x] Algoritmos GREEN, OMIT, POS_WANG e LMS em `algoritmos/`
- [x] Filtros no Processor: detrend + Butterworth bandpass [0.75–4.0 Hz]
- [x] HR em tempo real (a cada 30 frames, janela mínima 30 s)
- [x] `estimate_hr(bvp, fs)` → BPM via FFT (range 45–240 bpm)
- [x] Algoritmo LMS com IMU para cancelamento de artefactos de movimento
- [ ] Compensação de movimento via IMU nos algoritmos GREEN / OMIT / POS
- [ ] Validação clínica da estimativa HR vs. referência
