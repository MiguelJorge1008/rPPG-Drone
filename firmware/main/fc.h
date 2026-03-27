#pragma once

#include <stdint.h>
#include <stdbool.h>

/* ================================================================
 * fc.h — Flight Controller public API
 * ================================================================ */

typedef enum {
    FC_DISARMED = 0,   /* motors off                                */
    FC_FLIGHT,         /* full PID, accepts /cmd setpoints          */
} fc_state_t;

typedef struct {
    float roll;     /* degrees   [-30,  30]  */
    float pitch;    /* degrees   [-30,  30]  */
    float yaw;      /* deg/s     [-200, 200] */
    float thrust;   /* 0.0 – 1.0            */
} fc_setpoint_t;

typedef struct {
    float ax, ay, az;  /* g      */
    float gx, gy, gz;  /* deg/s  */
} fc_imu_t;

/**
 * Initialize motors (LEDC) + IMU (I2C) + stabilizer task.
 * Must be called before app_main starts the HTTP server.
 */
void fc_init(void);

/**
 * Update roll/pitch/yaw_rate/thrust setpoint.
 * Thread-safe. Also resets the watchdog timer.
 */
void fc_set_setpoint(const fc_setpoint_t *sp);

/** Transition to a new state. */
void fc_set_state(fc_state_t state);

fc_state_t  fc_get_state(void);

/** Latest estimated Euler angles (degrees). */
void fc_get_attitude(float *roll, float *pitch, float *yaw);

/** Latest raw IMU reading. */
void fc_get_imu(fc_imu_t *out);

/* ── PID configuration ─────────────────────────────────────── */
typedef struct {
    float att_kp;        /* attitude outer loop kp (roll = pitch) */
    float att_ki;        /* attitude outer loop ki                */
    float rate_roll_kp,  rate_roll_ki,  rate_roll_kd;
    float rate_pitch_kp, rate_pitch_ki, rate_pitch_kd;
    float rate_yaw_kp,   rate_yaw_ki,   rate_yaw_kd;
} fc_pid_cfg_t;

/** Update PID gains at runtime. Thread-safe. */
void fc_set_pid(const fc_pid_cfg_t *cfg);

/** Read current PID gains. */
void fc_get_pid(fc_pid_cfg_t *cfg);
