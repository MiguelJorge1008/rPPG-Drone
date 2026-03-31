"""
ADAPTIVE-LMS — Adaptive Motion Artifact Removal
Widrow & Hoff (1960) adaptive noise cancellation.

The GREEN channel is used as the noise-contaminated input signal.
The IMU (6 axes × L taps) serves as the noise reference signal.
The NLMS filter learns the weights in real time without prior training.

The tap delay line (L > 1) gives the filter temporal memory, allowing it
to cancel motion artefacts that span multiple frames — which single-sample
filtering cannot do.
"""

import numpy as np


def ADAPTIVE_LMS(rgb, imu_signal, mu=0.1, eps=1e-3, L=8):
    """
    Removes motion artifacts from the GREEN channel using NLMS adaptive
    filtering with IMU data as the noise reference signal.

    A tap delay line of length L is used so the filter has temporal memory:
    at each frame the input vector x contains the current and L-1 previous
    IMU samples across all 6 axes (shape 6*L), enabling cancellation of
    motion artefacts that span multiple frames.

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
        NLMS step size (learning rate). Default 0.1.
        Stable range: (0, 2). Higher = faster convergence, less stability.
    eps : float
        Regularisation term to avoid division by zero. Default 1e-3.
        Prevents step-size explosion when the drone is near-stationary.
    L : int
        Tap delay line length (number of past IMU frames per axis).
        Default 8 (~0.5 s at 15 fps). Higher = more temporal context,
        slower convergence.

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

    # NLMS with tap delay line
    # x[t] = [imu[t], imu[t-1], ..., imu[t-L+1]]  shape (6*L,)
    # noise_estimate = w · x
    # e = green - noise_estimate        (cleaned signal)
    # w += (mu / (x·x + eps)) * e * x  (weight update)
    w   = np.zeros(6 * L)
    buf = np.zeros((L, 6))   # delay buffer, newest sample at row 0
    output = np.zeros(N)

    for t in range(N):
        buf[1:] = buf[:-1]        # shift: row i becomes row i+1
        buf[0]  = imu_matrix[t]   # insert newest sample at front
        x = buf.flatten()         # shape (6*L,)
        noise_estimate = w @ x
        e = green[t] - noise_estimate
        w += (mu / (x @ x + eps)) * e * x
        output[t] = e

    return output
