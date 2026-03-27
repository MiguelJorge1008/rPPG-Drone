"""
POS — Plane-Orthogonal-to-Skin
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G.
Algorithmic principles of remote PPG.
IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491 (2017).
"""

import math
import numpy as np


def POS_WANG(rgb, fs):
    """
    Extracts the BVP signal using the POS sliding-window algorithm.

    Parameters
    ----------
    rgb : np.ndarray, shape (N, 3)
        Mean RGB values per frame, columns ordered [R, G, B].
    fs : float
        Sampling frequency in Hz (= video fps).
        Temporary default: 27. Replace with cap.get(cv2.CAP_PROP_FPS).

    Returns
    -------
    bvp : np.ndarray, shape (N,)
        Raw BVP signal (unfiltered).
    """
    WinSec = 1.6                          # window length in seconds (paper: l=32 @ 20fps)
    N = rgb.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec * fs)            # window length in frames

    for n in range(N):
        m = n - l
        if m >= 0:
            # Temporal normalisation (divide each channel by its window mean)
            Cn = np.true_divide(rgb[m:n, :], np.mean(rgb[m:n, :], axis=0))
            Cn = np.asmatrix(Cn).H        # conjugate transpose → (3, l)

            # Projection onto plane orthogonal to skin tone
            S = np.array([[0, 1, -1], [-2, 1, 1]]) @ Cn   # (2, l)

            # Alpha tuning (guard against zero std in S[1])
            std1 = np.std(S[1, :])
            alpha = np.std(S[0, :]) / std1 if std1 > 1e-9 else 0.0
            h = S[0, :] + alpha * S[1, :]
            h = np.asarray(h).flatten()
            h -= np.mean(h)

            H[m:n] += h

    return H
