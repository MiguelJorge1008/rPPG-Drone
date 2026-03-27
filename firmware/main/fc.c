/* ================================================================
 * fc.c — Flight Controller
 *
 * Sensor fusion : Mahony AHRS
 *                 (ported from sensfusion6.c — Crazyflie / ESP-Drone)
 * PID control   : cascade rate / attitude
 *                 (ported from pid.c         — Crazyflie / ESP-Drone)
 * Hardware      : LEDC PWM brushed motors + MPU-6050 @ 500 Hz
 *
 * Adapted from ESP-Drone (Espressif, GPL-3.0)
 *   Copyright 2019-2020 Espressif Systems (Shanghai)
 *   Copyright (C) 2011-2012 Bitcraze AB
 * ================================================================ */

#include "fc.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/ledc.h"
#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "FC";

/* ── Motor GPIOs ─────────────────────────────────────────────
 *  M1 front-left  (CW)  | M2 front-right (CCW)
 *  M4 back-left   (CCW) | M3 back-right  (CW)
 * ─────────────────────────────────────────────────────────── */
#define MOTOR_GPIO_M1   2
#define MOTOR_GPIO_M2   3
#define MOTOR_GPIO_M3   4
#define MOTOR_GPIO_M4   7

/* LEDC — TIMER_1 because TIMER_0 is used by the camera XCLK */
#define MOT_SPEED       LEDC_LOW_SPEED_MODE
#define MOT_TIMER       LEDC_TIMER_1
#define MOT_FREQ_HZ     15000
#define MOT_BITS        LEDC_TIMER_8_BIT
#define MOT_MAX_DUTY    255

/* ── IMU (MPU-6050) ──────────────────────────────────────────
 *  Same pins as in the original firmware (GPIO 5 = SCL, 6 = SDA).
 *  fc_init() takes ownership of the I2C bus.
 * ─────────────────────────────────────────────────────────── */
#define IMU_SCL         GPIO_NUM_5
#define IMU_SDA         GPIO_NUM_6
#define MPU_ADDR        0x68

/* MPU-6050 register map (subset) */
#define MPU_PWR_MGMT_1  0x6B
#define MPU_SMPLRT_DIV  0x19
#define MPU_CONFIG_REG  0x1A
#define MPU_GYRO_CFG    0x1B
#define MPU_ACCEL_CFG   0x1C
#define MPU_ACCEL_XOUT  0x3B

/* Full-scale ranges configured in fc_init() */
#define GYRO_SCALE      16.4f    /* ±2000 °/s  → 16.4  LSB/°/s  */
#define ACCEL_SCALE     2048.0f  /* ±16 g       → 2048  LSB/g    */

/* ── Timing ──────────────────────────────────────────────────*/
#define FC_HZ           500
#define FC_DT           (1.0f / FC_HZ)
#define FC_PERIOD_MS    (1000 / FC_HZ)   /* 2 ms                  */
#define FC_WARMUP_S     2                /* seconds before arming */

/* ── Safety ──────────────────────────────────────────────────*/
#define WDT_TIMEOUT_US  500000ULL        /* 500 ms watchdog        */
#define TILT_LIMIT_DEG  50.0f

/* ── Mahony AHRS gains (from sensfusion6.c) ──────────────────*/
#define TWO_KP          (2.0f * 0.4f)
#define TWO_KI          (2.0f * 0.001f)
#define M_PI_F          3.14159265f

/* ================================================================
 * Minimal self-contained PID
 * (algorithm from pid.c — Crazyflie, simplified: no D-filter)
 * ================================================================ */
typedef struct {
    float kp, ki, kd;
    float desired;
    float error, prev_error;
    float integ;
    float i_limit;
    float out_limit;
} pid_t;

static float pid_update(pid_t *p, float measured, float dt)
{
    p->error = p->desired - measured;
    float deriv = (p->error - p->prev_error) / dt;

    p->integ += p->error * dt;
    if (p->i_limit > 0.0f) {
        if (p->integ >  p->i_limit) p->integ =  p->i_limit;
        if (p->integ < -p->i_limit) p->integ = -p->i_limit;
    }

    float out = p->kp * p->error + p->ki * p->integ + p->kd * deriv;
    if (p->out_limit > 0.0f) {
        if (out >  p->out_limit) out =  p->out_limit;
        if (out < -p->out_limit) out = -p->out_limit;
    }

    p->prev_error = p->error;
    return out;
}

static void pid_reset(pid_t *p)
{
    p->error = p->prev_error = p->integ = 0.0f;
}

/* ================================================================
 * Global state
 * ================================================================ */
static volatile fc_state_t  s_state    = FC_DISARMED;
static fc_setpoint_t        s_sp       = {0};
static int64_t              s_last_sp_us = 0;
static SemaphoreHandle_t    s_sp_mutex;

/* Attitude + IMU data shared with HTTP handlers */
static float                s_roll = 0, s_pitch = 0, s_yaw = 0;
static fc_imu_t             s_imu  = {0};
static SemaphoreHandle_t    s_att_mutex;

/* ── Mahony quaternion state (from sensfusion6.c) ──────────── */
static float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
static float iFBx = 0.0f, iFBy = 0.0f, iFBz = 0.0f;

/* ── PID controllers ─────────────────────────────────────────
 * Cascade: attitude outer loop → rate inner loop
 * Gains are conservative starting values — tune on the bench.
 * ─────────────────────────────────────────────────────────── */
/* Outer: desired angle (°) → rate setpoint (°/s) */
static pid_t pid_att_roll  = { .kp=3.5f, .ki=2.0f, .kd=0.0f,
                                .i_limit=20.0f, .out_limit=20.0f };
static pid_t pid_att_pitch = { .kp=3.5f, .ki=2.0f, .kd=0.0f,
                                .i_limit=20.0f, .out_limit=20.0f };

/* Inner: desired rate (°/s) → torque output (raw) */
static pid_t pid_rate_roll  = { .kp=70.0f, .ki=0.0f, .kd=0.0f,
                                 .i_limit=33.3f, .out_limit=32767.0f };
static pid_t pid_rate_pitch = { .kp=70.0f, .ki=0.0f, .kd=0.0f,
                                 .i_limit=33.3f, .out_limit=32767.0f };
static pid_t pid_rate_yaw   = { .kp=50.0f, .ki=0.0f, .kd=0.0f,
                                 .i_limit=33.3f, .out_limit=32767.0f };

/* ── I2C handles ─────────────────────────────────────────────*/
static i2c_master_bus_handle_t s_i2c_bus;
static i2c_master_dev_handle_t s_mpu_dev;

/* ================================================================
 * Private helpers
 * ================================================================ */

/* Fast inverse square-root (from sensfusion6.c / Quake III) */
static float inv_sqrt(float x)
{
    float halfx = 0.5f * x;
    float y = x;
    int32_t i;
    memcpy(&i, &y, sizeof(i));
    i = 0x5f3759df - (i >> 1);
    memcpy(&y, &i, sizeof(y));
    y = y * (1.5f - (halfx * y * y));
    return y;
}

/* Mahony AHRS update (from sensfusion6.c — Crazyflie, Mahony variant) */
static void mahony_update(float gx, float gy, float gz,
                          float ax, float ay, float az, float dt)
{
    float recipNorm;
    float halfvx, halfvy, halfvz;
    float halfex, halfey, halfez;
    float qa, qb, qc;

    /* Convert gyro from deg/s to rad/s */
    gx *= M_PI_F / 180.0f;
    gy *= M_PI_F / 180.0f;
    gz *= M_PI_F / 180.0f;

    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
        recipNorm = inv_sqrt(ax*ax + ay*ay + az*az);
        ax *= recipNorm; ay *= recipNorm; az *= recipNorm;

        /* Estimated gravity direction from current quaternion */
        halfvx = qx*qz - qw*qy;
        halfvy = qw*qx + qy*qz;
        halfvz = qw*qw - 0.5f + qz*qz;

        /* Error: cross product of estimated and measured gravity */
        halfex = (ay * halfvz - az * halfvy);
        halfey = (az * halfvx - ax * halfvz);
        halfez = (ax * halfvy - ay * halfvx);

        /* Integral feedback */
        if (TWO_KI > 0.0f) {
            iFBx += TWO_KI * halfex * dt;
            iFBy += TWO_KI * halfey * dt;
            iFBz += TWO_KI * halfez * dt;
            gx += iFBx; gy += iFBy; gz += iFBz;
        } else {
            iFBx = iFBy = iFBz = 0.0f;
        }

        /* Proportional feedback */
        gx += TWO_KP * halfex;
        gy += TWO_KP * halfey;
        gz += TWO_KP * halfez;
    }

    /* Integrate quaternion rate */
    gx *= 0.5f * dt; gy *= 0.5f * dt; gz *= 0.5f * dt;
    qa = qw; qb = qx; qc = qy;
    qw += (-qb*gx - qc*gy - qz*gz);
    qx += ( qa*gx + qc*gz - qz*gy);
    qy += ( qa*gy - qb*gz + qz*gx);
    qz += ( qa*gz + qb*gy - qc*gx);

    /* Normalise */
    recipNorm = inv_sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw *= recipNorm; qx *= recipNorm; qy *= recipNorm; qz *= recipNorm;
}

/* Convert quaternion → Euler angles in degrees (from sensfusion6.c) */
static void quat_to_euler(float *roll, float *pitch, float *yaw)
{
    float gx = 2.0f * (qx*qz - qw*qy);
    float gy = 2.0f * (qw*qx + qy*qz);
    float gz = qw*qw - qx*qx - qy*qy + qz*qz;

    if (gx >  1.0f) gx =  1.0f;
    if (gx < -1.0f) gx = -1.0f;

    *roll  = atan2f(gy, gz) * 180.0f / M_PI_F;
    *pitch = asinf(gx)      * 180.0f / M_PI_F;
    *yaw   = atan2f(2.0f*(qw*qz + qx*qy),
                    qw*qw + qx*qx - qy*qy - qz*qz) * 180.0f / M_PI_F;
}

/* Set all four motor duties in a single call */
static void motors_set(uint8_t m1, uint8_t m2, uint8_t m3, uint8_t m4)
{
    ledc_set_duty(MOT_SPEED, LEDC_CHANNEL_1, m1);
    ledc_update_duty(MOT_SPEED, LEDC_CHANNEL_1);
    ledc_set_duty(MOT_SPEED, LEDC_CHANNEL_2, m2);
    ledc_update_duty(MOT_SPEED, LEDC_CHANNEL_2);
    ledc_set_duty(MOT_SPEED, LEDC_CHANNEL_3, m3);
    ledc_update_duty(MOT_SPEED, LEDC_CHANNEL_3);
    ledc_set_duty(MOT_SPEED, LEDC_CHANNEL_4, m4);
    ledc_update_duty(MOT_SPEED, LEDC_CHANNEL_4);
}

static void motors_off(void)
{
    motors_set(0, 0, 0, 0);
}

static inline int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Write one register to MPU-6050 */
static void mpu_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };
    i2c_master_transmit(s_mpu_dev, buf, 2, 10);
}

/* Read accel + gyro (14 bytes: accel[6] + temp[2] + gyro[6]) */
static esp_err_t mpu_read(int16_t *ax, int16_t *ay, int16_t *az,
                           int16_t *gx, int16_t *gy, int16_t *gz)
{
    uint8_t reg = MPU_ACCEL_XOUT;
    uint8_t buf[14];
    esp_err_t err = i2c_master_transmit_receive(s_mpu_dev, &reg, 1, buf, 14, 10);
    if (err != ESP_OK) return err;

    *ax = (int16_t)((buf[0]  << 8) | buf[1]);
    *ay = (int16_t)((buf[2]  << 8) | buf[3]);
    *az = (int16_t)((buf[4]  << 8) | buf[5]);
    /* buf[6..7] = temperature, skip */
    *gx = (int16_t)((buf[8]  << 8) | buf[9]);
    *gy = (int16_t)((buf[10] << 8) | buf[11]);
    *gz = (int16_t)((buf[12] << 8) | buf[13]);
    return ESP_OK;
}

/* ================================================================
 * Stabilizer task — runs on Core 1 at 500 Hz, priority 8
 * ================================================================ */
static void fc_task(void *arg)
{
    TickType_t last_wake = xTaskGetTickCount();
    int warmup_remaining = FC_HZ * FC_WARMUP_S;

    while (1) {
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(FC_PERIOD_MS));

        /* ── 1. Read IMU ─────────────────────────────────────── */
        int16_t rax, ray, raz, rgx, rgy, rgz;
        if (mpu_read(&rax, &ray, &raz, &rgx, &rgy, &rgz) != ESP_OK) {
            motors_off();
            continue;
        }

        /* Axes remapped to match physical drone orientation:
         * IMU X→drone Y (pitch), IMU Y→drone X (roll) */
        float ax =  ray / ACCEL_SCALE;
        float ay =  rax / ACCEL_SCALE;
        float az =  raz / ACCEL_SCALE;
        float gx =  rgy / GYRO_SCALE;
        float gy =  rgx / GYRO_SCALE;
        float gz =  rgz / GYRO_SCALE;

        /* ── 2. Mahony AHRS ──────────────────────────────────── */
        mahony_update(gx, gy, gz, ax, ay, az, FC_DT);

        float roll, pitch, yaw;
        quat_to_euler(&roll, &pitch, &yaw);

        /* ── 3. Publish attitude + IMU data ──────────────────── */
        xSemaphoreTake(s_att_mutex, portMAX_DELAY);
        s_roll  = roll;
        s_pitch = pitch;
        s_yaw   = yaw;
        s_imu.ax = ax; s_imu.ay = ay; s_imu.az = az;
        s_imu.gx = gx; s_imu.gy = gy; s_imu.gz = gz;
        xSemaphoreGive(s_att_mutex);

        /* ── 4. Warmup (Mahony convergence) ──────────────────── */
        if (warmup_remaining > 0) {
            warmup_remaining--;
            motors_off();
            continue;
        }

        /* ── 5. State check ──────────────────────────────────── */
        fc_state_t state = s_state;

        if (state == FC_DISARMED) {
            motors_off();
            pid_reset(&pid_att_roll);   pid_reset(&pid_att_pitch);
            pid_reset(&pid_rate_roll);  pid_reset(&pid_rate_pitch);
            pid_reset(&pid_rate_yaw);
            continue;
        }

        /* ── 6. Tilt protection ──────────────────────────────── */
        if (fabsf(roll) > TILT_LIMIT_DEG || fabsf(pitch) > TILT_LIMIT_DEG) {
            ESP_LOGW(TAG, "TILT PROTECTION (roll=%.1f pitch=%.1f) — disarming",
                     roll, pitch);
            s_state = FC_DISARMED;
            motors_off();
            continue;
        }

        /* ── 7. Watchdog (FLIGHT mode only) ──────────────────── */
        if (state == FC_FLIGHT) {
            xSemaphoreTake(s_sp_mutex, portMAX_DELAY);
            int64_t age = esp_timer_get_time() - s_last_sp_us;
            xSemaphoreGive(s_sp_mutex);
            if (age > (int64_t)WDT_TIMEOUT_US) {
                ESP_LOGW(TAG, "WDT timeout (%" PRId64 " ms) — disarming", age / 1000);
                s_state = FC_DISARMED;
                motors_off();
                continue;
            }
        }

        /* ── 8. Read setpoint ────────────────────────────────── */
        xSemaphoreTake(s_sp_mutex, portMAX_DELAY);
        fc_setpoint_t sp = s_sp;
        xSemaphoreGive(s_sp_mutex);

        /* ── 9. Cascade PID ──────────────────────────────────── */
        /* Outer loop: angle setpoint → rate setpoint */
        pid_att_roll.desired  = sp.roll;
        pid_att_pitch.desired = sp.pitch;
        float rate_sp_roll  = pid_update(&pid_att_roll,  roll,  FC_DT);
        float rate_sp_pitch = pid_update(&pid_att_pitch, pitch, FC_DT);

        /* Inner loop: rate setpoint → torque output */
        pid_rate_roll.desired  = rate_sp_roll;
        pid_rate_pitch.desired = rate_sp_pitch;
        pid_rate_yaw.desired   = sp.yaw;
        float out_roll  = pid_update(&pid_rate_roll,  gx, FC_DT);
        float out_pitch = pid_update(&pid_rate_pitch, gy, FC_DT);
        float out_yaw   = pid_update(&pid_rate_yaw,   gz, FC_DT);

        /* ── 10. Motor mixing (X frame) ──────────────────────── */
        /* Scale PID torque output [-32767, 32767] → motor delta [-127, 127] */
        int base    = (int)(sp.thrust * MOT_MAX_DUTY);
        int roll_m  = (int)(out_roll  / 32767.0f * 127.0f);
        int pitch_m = (int)(out_pitch / 32767.0f * 127.0f);
        int yaw_m   = (int)(out_yaw   / 32767.0f * 127.0f);

        /*
         * X frame:
         *   M1 (FL, CW) : +roll +pitch -yaw
         *   M2 (FR, CCW): -roll +pitch +yaw
         *   M3 (BR, CW) : -roll -pitch -yaw
         *   M4 (BL, CCW): +roll -pitch +yaw
         */
        int m1 = clamp_int(base + roll_m + pitch_m - yaw_m, 0, MOT_MAX_DUTY);
        int m2 = clamp_int(base - roll_m + pitch_m + yaw_m, 0, MOT_MAX_DUTY);
        int m3 = clamp_int(base - roll_m - pitch_m - yaw_m, 0, MOT_MAX_DUTY);
        int m4 = clamp_int(base + roll_m - pitch_m + yaw_m, 0, MOT_MAX_DUTY);

        motors_set((uint8_t)m1, (uint8_t)m2, (uint8_t)m3, (uint8_t)m4);
    }
}

/* ================================================================
 * Public API
 * ================================================================ */

void fc_init(void)
{
    s_sp_mutex  = xSemaphoreCreateMutex();
    s_att_mutex = xSemaphoreCreateMutex();

    /* ── Motor LEDC (TIMER_1, channels 1-4) ─────────────────── */
    ledc_timer_config_t ledc_timer = {
        .speed_mode      = MOT_SPEED,
        .timer_num       = MOT_TIMER,
        .duty_resolution = MOT_BITS,
        .freq_hz         = MOT_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));

    const int motor_gpios[4] = {
        MOTOR_GPIO_M1, MOTOR_GPIO_M2, MOTOR_GPIO_M3, MOTOR_GPIO_M4
    };
    for (int i = 0; i < 4; i++) {
        gpio_config_t io = {
            .pin_bit_mask = (1ULL << motor_gpios[i]),
            .mode         = GPIO_MODE_OUTPUT,
            .pull_down_en = GPIO_PULLDOWN_ENABLE,
            .pull_up_en   = GPIO_PULLUP_DISABLE,
            .intr_type    = GPIO_INTR_DISABLE,
        };
        gpio_config(&io);
        gpio_set_level(motor_gpios[i], 0);
    }
    for (int i = 0; i < 4; i++) {
        ledc_channel_config_t ch = {
            .channel    = (ledc_channel_t)(LEDC_CHANNEL_1 + i),
            .duty       = 0,
            .gpio_num   = motor_gpios[i],
            .speed_mode = MOT_SPEED,
            .timer_sel  = MOT_TIMER,
            .hpoint     = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch));
    }
    ESP_LOGI(TAG, "Motors OK — GPIO %d %d %d %d | LEDC_TIMER_1 | 15 kHz | 8-bit",
             MOTOR_GPIO_M1, MOTOR_GPIO_M2, MOTOR_GPIO_M3, MOTOR_GPIO_M4);

    /* ── I2C bus + MPU-6050 ──────────────────────────────────── */
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port                = I2C_NUM_0,
        .sda_io_num              = IMU_SDA,
        .scl_io_num              = IMU_SCL,
        .clk_source              = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt       = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &s_i2c_bus));

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address  = MPU_ADDR,
        .scl_speed_hz    = 400000,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(s_i2c_bus, &dev_cfg, &s_mpu_dev));

    /* Configure MPU-6050 for flight */
    mpu_write_reg(MPU_PWR_MGMT_1, 0x01);   /* PLL with X gyro reference */
    vTaskDelay(pdMS_TO_TICKS(100));
    mpu_write_reg(MPU_SMPLRT_DIV,  0x01);  /* Sample rate = 1 kHz / 2 = 500 Hz */
    mpu_write_reg(MPU_CONFIG_REG,  0x02);  /* DLPF: ~100 Hz BW, ~2 ms delay    */
    mpu_write_reg(MPU_GYRO_CFG,    0x18);  /* ±2000 °/s                         */
    mpu_write_reg(MPU_ACCEL_CFG,   0x18);  /* ±16 g                             */
    vTaskDelay(pdMS_TO_TICKS(10));
    ESP_LOGI(TAG, "MPU-6050 OK — 500 Hz | ±2000 °/s | ±16 g");

    /* Stabilizer task pinned to Core 1 (Wi-Fi runs on Core 0) */
    xTaskCreatePinnedToCore(fc_task, "fc_stab", 4096, NULL, 8, NULL, 1);
    ESP_LOGI(TAG, "Stabilizer task started — Core 1 | priority 8 | 500 Hz");
}

void fc_set_setpoint(const fc_setpoint_t *sp)
{
    xSemaphoreTake(s_sp_mutex, portMAX_DELAY);
    s_sp         = *sp;
    s_last_sp_us = esp_timer_get_time();
    xSemaphoreGive(s_sp_mutex);
}

void fc_set_state(fc_state_t state)
{
    if (state != FC_DISARMED) {
        /* Reset watchdog on arm/hover */
        xSemaphoreTake(s_sp_mutex, portMAX_DELAY);
        s_last_sp_us = esp_timer_get_time();
        xSemaphoreGive(s_sp_mutex);
    }
    s_state = state;
    ESP_LOGI(TAG, "State → %s",
             state == FC_DISARMED      ? "DISARMED"      :
             state == FC_FLIGHT        ? "FLIGHT"        :
                                         "HOVER_ACQUIRE");
}

fc_state_t fc_get_state(void)
{
    return s_state;
}

void fc_get_attitude(float *roll, float *pitch, float *yaw)
{
    xSemaphoreTake(s_att_mutex, portMAX_DELAY);
    *roll  = s_roll;
    *pitch = s_pitch;
    *yaw   = s_yaw;
    xSemaphoreGive(s_att_mutex);
}

void fc_get_imu(fc_imu_t *out)
{
    xSemaphoreTake(s_att_mutex, portMAX_DELAY);
    *out = s_imu;
    xSemaphoreGive(s_att_mutex);
}

void fc_set_pid(const fc_pid_cfg_t *cfg)
{
    xSemaphoreTake(s_sp_mutex, portMAX_DELAY);
    pid_att_roll.kp  = pid_att_pitch.kp  = cfg->att_kp;
    pid_att_roll.ki  = pid_att_pitch.ki  = cfg->att_ki;
    pid_rate_roll.kp  = cfg->rate_roll_kp;
    pid_rate_roll.ki  = cfg->rate_roll_ki;
    pid_rate_roll.kd  = cfg->rate_roll_kd;
    pid_rate_pitch.kp = cfg->rate_pitch_kp;
    pid_rate_pitch.ki = cfg->rate_pitch_ki;
    pid_rate_pitch.kd = cfg->rate_pitch_kd;
    pid_rate_yaw.kp   = cfg->rate_yaw_kp;
    pid_rate_yaw.ki   = cfg->rate_yaw_ki;
    pid_rate_yaw.kd   = cfg->rate_yaw_kd;
    xSemaphoreGive(s_sp_mutex);
}

void fc_get_pid(fc_pid_cfg_t *cfg)
{
    xSemaphoreTake(s_sp_mutex, portMAX_DELAY);
    cfg->att_kp        = pid_att_roll.kp;
    cfg->att_ki        = pid_att_roll.ki;
    cfg->rate_roll_kp  = pid_rate_roll.kp;
    cfg->rate_roll_ki  = pid_rate_roll.ki;
    cfg->rate_roll_kd  = pid_rate_roll.kd;
    cfg->rate_pitch_kp = pid_rate_pitch.kp;
    cfg->rate_pitch_ki = pid_rate_pitch.ki;
    cfg->rate_pitch_kd = pid_rate_pitch.kd;
    cfg->rate_yaw_kp   = pid_rate_yaw.kp;
    cfg->rate_yaw_ki   = pid_rate_yaw.ki;
    cfg->rate_yaw_kd   = pid_rate_yaw.kd;
    xSemaphoreGive(s_sp_mutex);
}
