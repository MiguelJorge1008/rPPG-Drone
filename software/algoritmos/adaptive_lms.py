"""
ADAPTIVE-LMS — Adaptive Motion Artifact Removal
Widrow & Hoff (1960) adaptive noise cancellation.

The GREEN channel is used as the noise-contaminated input signal.
The IMU (6 axes) serves as the noise reference signal.
The NLMS filter learns the weights in real time without prior training.
"""

import numpy as np


def ADAPTIVE_LMS(rgb, imu_signal, mu=0.01, eps=1e-6):
    """
    Removes motion artifacts from the GREEN channel using NLMS adaptive
    filtering with IMU data as the noise reference signal.

    The IMU signal is interpolated from its native sampling rate to match
    the rPPG frame rate before filtering.

    Parameters
    ----------
    rgb : np.ndarray, shape (N, 3)
        Mean RGB values per frame, columns ordered [R, G, B].
    imu_signal : list of dict
        IMU samples with keys ax, ay, az (g), gx, gy, gz (deg/s).
        Length M may differ from N.
    mu : float
        NLMS step size (learning rate). Default 0.01.
    eps : float
        Regularisation term to avoid division by zero. Default 1e-6.

    Returns
    -------
    bvp : np.ndarray, shape (N,)
        Motion-corrected BVP signal (unfiltered).
    """
    green = rgb[:, 1].astype(np.float64)
    N = len(green)
    M = len(imu_signal)

    # Interpolate IMU to match the rPPG frame count
    t_imu  = np.linspace(0, 1, M)
    t_rppg = np.linspace(0, 1, N)
    fields = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    imu_matrix = np.zeros((N, 6))
    for i, field in enumerate(fields):
        values = np.array([s[field] for s in imu_signal], dtype=np.float64)
        imu_matrix[:, i] = np.interp(t_rppg, t_imu, values)

    # NLMS — Normalized Least Mean Squares
    # At each frame:
    #   noise_estimate = w · x
    #   e = green - noise_estimate        (cleaned signal)
    #   w += (mu / (x·x + eps)) * e * x  (weight update)
    w = np.zeros(6)
    output = np.zeros(N)
    for t in range(N):
        x = imu_matrix[t]
        noise_estimate = w @ x
        e = green[t] - noise_estimate
        w += (mu / (x @ x + eps)) * e * x
        output[t] = e

    return output
