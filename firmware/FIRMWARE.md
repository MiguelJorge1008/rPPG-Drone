# Firmware — Drone rPPG (XIAO ESP32-S3 Sense)

Firmware para captura de vídeo MJPEG via Wi-Fi com sensor OV3660 e controlo de voo com motores brushed, optimizado para **rPPG (Remote Photoplethysmography)** embarcado em drone.

---

## Hardware

| Componente | Modelo |
|---|---|
| MCU | Seeed Studio XIAO ESP32-S3 Sense |
| Câmara | OV3660 (QQVGA 160×120, ~25 FPS, MJPEG) |
| IMU | MPU-6050 (I2C, 500 Hz, ±16 g / ±2000°/s) |
| Motores | 4× brushed DC (LEDC PWM 15 kHz, 8-bit) |
| Bateria | LiPo 1S (monitorização ADC GPIO 1) |
| PSRAM | 8 MB (Octal) |
| Flash | 8 MB |
| Framework | ESP-IDF v5.5.3 |

---

## Modo de rede

O firmware opera em modo **Access Point (AP)** — o ESP32 cria a sua própria rede Wi-Fi.

| Parâmetro | Valor |
|---|---|
| SSID | `rPPG-Drone` |
| Password | `12345678` |
| IP fixo | `192.168.4.1` |
| Canal | 2 (2,4 GHz) |
| Segurança | WPA2-PSK |

---

## Configuração rápida

1. Build e flash:
```bash
idf.py build flash monitor
```

2. Ligar o PC/telemóvel à rede `rPPG-Drone` (password: `12345678`)

3. Abrir browser em `http://192.168.4.1`

---

## Endpoints HTTP

Existem **dois servidores HTTP**:
- **Porta 80** — página HTML, controlos, IMU, voo
- **Porta 81** — stream MJPEG (servidor dedicado, não bloqueia os outros endpoints)

| URL | Método | Servidor | Descrição |
|---|---|---|---|
| `/` | GET | :80 | Página HTML com stream + controlos de câmara, voo e PID |
| `/stream` | GET | :81 | Stream MJPEG contínuo |
| `/control?aec=X&gain=Y` | GET | :80 | Ajusta exposição e ganho em tempo real |
| `/imu` | GET | :80 | JSON com IMU raw, atitude estimada, bateria e estado FC |
| `/cmd?state=arm\|disarm` | GET | :80 | Armar/desarmar o flight controller |
| `/cmd?roll=&pitch=&yaw=&thrust=` | GET | :80 | Setpoint de voo manual |
| `/pid[?att_kp=&...]` | GET | :80 | Ler ou actualizar ganhos PID em runtime |
| `/ws` | WS | :80 | WebSocket para setpoints de joystick a 20 Hz |

### Parâmetros de `/control`

| Parâmetro | Range | Descrição |
|---|---|---|
| `aec` | 0 – 1200 | Tempo de exposição manual (OV3660) |
| `gain` | 0 – 30 | Ganho analógico manual |

### Resposta `/imu`

```json
{
  "ax": 0.01, "ay": -0.02, "az": 0.99,
  "gx": 10.5, "gy": -5.2, "gz": 0.1,
  "roll": 1.2, "pitch": -0.8, "yaw": 45.0,
  "bat": 3850,
  "fc_state": "disarmed"
}
```

| Campo | Unidade | Descrição |
|---|---|---|
| `ax`, `ay`, `az` | g | Aceleração linear (±16 g, 2048 LSB/g) |
| `gx`, `gy`, `gz` | deg/s | Velocidade angular (±2000°/s, 16.4 LSB/°/s) |
| `roll`, `pitch`, `yaw` | graus | Atitude estimada pelo Mahony AHRS |
| `bat` | mV | Tensão da bateria (divisor 100K+100K, GPIO1) |
| `fc_state` | string | `"disarmed"` ou `"flight"` |

### Parâmetros de `/cmd`

| Parâmetro | Range | Descrição |
|---|---|---|
| `state` | `arm` / `disarm` | Transição de estado do FC |
| `roll` | −30 – +30 | Ângulo de roll desejado (graus) |
| `pitch` | −30 – +30 | Ângulo de pitch desejado (graus) |
| `yaw` | −200 – +200 | Velocidade de yaw desejada (°/s) |
| `thrust` | 0.0 – 1.0 | Impulso base normalizado |

### Parâmetros de `/pid`

GET sem query → devolve JSON com os ganhos actuais.
GET com query → aplica os novos ganhos e devolve os valores confirmados.

| Parâmetro | Descrição | Default |
|---|---|---|
| `att_kp` | Outer loop kp (roll = pitch) | 3.5 |
| `att_ki` | Outer loop ki | 2.0 |
| `rr_kp/ki/kd` | Rate roll kp/ki/kd | 70 / 0 / 0 |
| `rp_kp/ki/kd` | Rate pitch kp/ki/kd | 70 / 0 / 0 |
| `ry_kp/ki/kd` | Rate yaw kp/ki/kd | 50 / 0 / 0 |

### WebSocket `/ws`

O cliente envia JSON a 20 Hz:
```json
{"r": 5.0, "p": -3.0, "y": 10.0, "t": 0.55}
```

| Campo | Unidade | Descrição |
|---|---|---|
| `r` | graus | Roll (−30 a +30) |
| `p` | graus | Pitch (−30 a +30) |
| `y` | °/s | Yaw rate (−200 a +200) |
| `t` | 0.0 – 1.0 | Thrust |

Cada frame WS reinicia o watchdog do FC. Se não chegarem frames durante **500 ms**, o FC passa automaticamente para `DISARMED`.

---

## Flight Controller

O flight controller está implementado em `main/fc.c` + `main/fc.h` e corre numa FreeRTOS task dedicada no **Core 1** a **500 Hz** (prioridade 8).

### Arquitectura

```
MPU-6050 (500 Hz)
    │
    ▼
Mahony AHRS  →  Euler angles (roll, pitch, yaw)
    │
    ▼
Outer PID (atitude)  →  rate setpoint
    │
    ▼
Inner PID (rate)  →  torques (roll, pitch, yaw)
    │
    ▼
Motor mixing  →  M1/M2/M3/M4 duties (0–255)
```

**Sensor fusion:** Mahony AHRS (portado de `sensfusion6.c` — Crazyflie / ESP-Drone).
**PID:** Cascade rate/atitude (portado de `pid.c` — Crazyflie / ESP-Drone).

### Estados do FC

| Estado | Descrição |
|---|---|
| `FC_DISARMED` | Motores desligados, PIDs resetados |
| `FC_FLIGHT` | PID activo, aceita setpoints via `/cmd` ou `/ws` |

### Segurança

| Mecanismo | Valor | Comportamento |
|---|---|---|
| Watchdog de setpoint | 500 ms | FC passa a DISARMED se não receber setpoint |
| Tilt protection | ±50° | FC passa a DISARMED se roll ou pitch exceder |
| Warmup Mahony | 2 s | Motores bloqueados no arranque para convergência AHRS |
| Bateria crítica | < 3100 mV | FC passa a DISARMED + deep sleep |

### Motor mixing (quadrotor X)

```
M1 (front-left, CW)  |  M2 (front-right, CCW)
─────────────────────────────────────────────
M4 (back-left, CCW)  |  M3 (back-right, CW)
```

| Motor | GPIO | Rotação | LEDC Channel |
|---|---|---|---|
| M1 (frente-esq) | GPIO 2 | CW | CH1 |
| M2 (frente-dir) | GPIO 3 | CCW | CH2 |
| M3 (trás-dir) | GPIO 4 | CW | CH3 |
| M4 (trás-esq) | GPIO 7 | CCW | CH4 |

**LEDC:** Timer 1, 15 kHz, 8-bit (0–255). O Timer 0 / Channel 0 está reservado ao XCLK da câmara.

---

## Bateria

| Parâmetro | Valor |
|---|---|
| Pin ADC | GPIO 1 (ADC1 CH0) |
| Divisor de tensão | 100 kΩ + 100 kΩ (fator ×2) |
| Aviso amarelo | < 3500 mV |
| Crítico / deep sleep | < 3100 mV |
| Sem bateria | < 2000 mV (ignora alarme) |
| LED status | GPIO 21 |

Quando a bateria entra em crítico: FC passa a DISARMED → LED pisca 10× → `esp_deep_sleep_start()`.

---

## Configuração do sensor OV3660 para rPPG

### Porquê desligar os automáticos

Para rPPG é **obrigatório** desligar todos os controlos automáticos do sensor. Se estiverem activos, o sensor compensa continuamente as variações de brilho na pele — que são precisamente o sinal que se quer capturar.

| Controlo | Função desligada | Razão |
|---|---|---|
| AWB (Auto White Balance) | `set_whitebal(s, 0)` | Reequilibraria os canais R/G/B frame a frame |
| AGC (Auto Gain Control) | `set_gain_ctrl(s, 0)` | Variações de ganho introduzem artefactos no sinal PPG |
| AEC (Auto Exposure) | `set_exposure_ctrl(s, 0)` | O AEC compensa exactamente as variações de reflectância da pele |
| AEC nightmode | `set_aec2(s, 0)` | Modo DSP extra do AEC, igualmente prejudicial |

### Valores por defeito calibrados

```c
s->set_agc_gain(s, 3);        // Ganho fixo
s->set_aec_value(s, 1200);    // Exposição fixa — ajustável em runtime via /control
```

| Condição | aec recomendado | gain |
|---|---|---|
| Exterior — sol directo | 20 – 60 | 0 – 2 |
| Exterior — nublado | 100 – 200 | 2 – 4 |
| Interior (teste) | 400 – 800 | 4 – 8 |

### Cor verde — comportamento esperado

Com AWB desligado, a imagem apresenta um **cast verde pronunciado**. Isto é normal:
- O sensor OV3660 usa padrão Bayer **RGGB** — 2 pixels verdes por cada 1 vermelho e 1 azul
- Sem correcção de cor automática, o verde domina (resposta raw do sensor)
- O canal verde é o **mais sensível ao sinal PPG** (~550 nm)
- A maioria dos algoritmos rPPG (GREEN, CHROM, POS) extrai o sinal principalmente do canal verde

### Flicker de iluminação em interior

Em ambiente indoor com lâmpadas LED/fluorescentes, podem aparecer variações periódicas de brilho. Isto deve-se ao pulso da rede eléctrica (100 Hz em Europa / 50 Hz AC).

- Se a variação ocorre em zonas não-pele → é flicker de iluminação
- Se a variação ocorre principalmente na pele de forma periódica (~60–80 bpm) → é sinal rPPG

Em exterior com luz solar este problema não existe.

---

## Performance

| Parâmetro | Valor |
|---|---|
| Resolução | **QQVGA (160×120)** |
| FPS médio | **~25 FPS** (dependente da rede Wi-Fi) |
| JPEG quality | 8 |
| Frame buffers | 3 em PSRAM (`CAMERA_GRAB_LATEST`) |
| XCLK | 24 MHz |
| Log série | `>>> STREAMING: XX.XX FPS \| Res: QVGA` (a cada segundo) |

---

## Servidor HTTP

```c
/* Porta 80 — página + controlos */
server_config.stack_size       = 8192;
server_config.max_uri_handlers = 10;
server_config.max_open_sockets = 7;

/* Porta 81 — stream MJPEG */
stream_config.stack_size       = 8192;
stream_config.max_uri_handlers = 2;
stream_config.max_open_sockets = 2;
```

O stream MJPEG corre num servidor separado (porta 81) para que pedidos ao `/control`, `/imu`, `/cmd`, etc. nunca fiquem bloqueados pelo socket TCP persistente do stream.

A página HTML pausa o stream antes de enviar o pedido `/control` e reconecta 200 ms depois — evita congelamento da imagem por contenção de sockets.

---

## Mapeamento de pinos (XIAO ESP32-S3 Sense)

### Câmara (OV3660)

| Sinal | GPIO |
|---|---|
| XCLK | 10 |
| SDA (SCCB) | 40 |
| SCL (SCCB) | 39 |
| D0–D7 | 15, 17, 18, 16, 14, 12, 11, 48 |
| VSYNC | 38 |
| HREF | 47 |
| PCLK | 13 |
| PWDN / RESET | N/A (-1) |

### IMU (MPU-6050)

| Sinal | GPIO |
|---|---|
| SCL (I2C) | 5 |
| SDA (I2C) | 6 |
| Endereço I2C | 0x68 |
| Sample rate | 500 Hz |
| Gyro full-scale | ±2000°/s (16.4 LSB/°/s) |
| Accel full-scale | ±16 g (2048 LSB/g) |

### Motores (LEDC PWM)

| Sinal | GPIO |
|---|---|
| M1 (frente-esq, CW) | 2 |
| M2 (frente-dir, CCW) | 3 |
| M3 (trás-dir, CW) | 4 |
| M4 (trás-esq, CCW) | 7 |

### Outros

| Sinal | GPIO |
|---|---|
| Bateria ADC | 1 (ADC1 CH0) |
| LED status | 21 |

---

## Troubleshooting

**O browser mostra apenas o stream sem controlos**
→ Cache da versão antiga. Fazer Ctrl+Shift+R ou abrir em modo anónimo.

**O stream não aparece (`http://192.168.4.1`)**
→ O stream está na porta 81. A página HTML aponta para `http://192.168.4.1:81/stream`. Verificar se o segundo servidor arrancou nos logs série.

**Compile time no monitor série é antigo**
→ O firmware não foi reflashado. Correr `idf.py build flash monitor`.

**Os controlos de câmara não alteram a imagem**
→ O JS pausa o stream, envia `/control`, reconecta. Verificar logs série: `Control: OK | aec=X gain=Y`.

**Saturação na testa / overexposure**
→ Reduzir `aec` via slider. Para exterior com sol, usar aec < 60.

**A rede `rPPG-Drone` não aparece**
→ Verificar no monitor série: `AP started: rPPG-Drone | 192.168.4.1`. Se não aparecer, reflashar.

**FC não arma (botão ARM sem efeito)**
→ Aguardar 2 s após boot (warmup Mahony). Verificar se o MPU-6050 inicializou sem erro nos logs série.

**FC desarma sozinho em voo**
→ Pode ser watchdog (sem setpoints há mais de 500 ms) ou tilt protection (inclinação > 50°). Verificar logs série: `WDT timeout` ou `TILT PROTECTION`.

**Sem acesso à internet enquanto ligado ao ESP32**
→ Comportamento esperado — o AP do ESP32 não fornece internet. Preparar tudo antes de mudar de rede.
