"""
GREEN
Verkruysse, W., Svaasand, L. O. & Nelson, J. S.
Remote plethysmographic imaging using ambient light.
Optical Express 16, 21434–21445 (2008).
"""

import numpy as np


def GREEN(rgb):
    """
    Extracts the BVP signal from the green channel.

    Parameters
    ----------
    rgb : np.ndarray, shape (N, 3)
        Mean RGB values per frame, columns ordered [R, G, B].

    Returns
    -------
    bvp : np.ndarray, shape (N,)
        Raw BVP signal (unfiltered).
    """
    return rgb[:, 1].copy()
