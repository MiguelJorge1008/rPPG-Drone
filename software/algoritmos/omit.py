"""
OMIT — Orthogonal Matrix Image Transformation
Álvarez Casado, C., & Bordallo López, M.
Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.
IEEE Journal of Biomedical and Health Informatics (2023).
"""

import numpy as np


def OMIT(rgb):
    """
    Extracts the BVP signal via QR decomposition of the RGB matrix.

    Parameters
    ----------
    rgb : np.ndarray, shape (N, 3)
        Mean RGB values per frame, columns ordered [R, G, B].

    Returns
    -------
    bvp : np.ndarray, shape (N,)
        Raw BVP signal (unfiltered).
    """
    # QR decomposition on the (3, N) matrix
    data = rgb.T                          # (3, N)
    Q, _ = np.linalg.qr(data)            # Q: (3, 3)
    S = Q[:, 0].reshape(1, -1)           # dominant direction, shape (1, 3)
    P = np.identity(3) - S.T @ S         # projection matrix (3, 3)
    Y = P @ data                          # project out dominant component (3, N)
    return Y[1, :]                        # second row as BVP signal
