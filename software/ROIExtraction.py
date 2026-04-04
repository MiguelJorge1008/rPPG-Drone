import cv2
import numpy as np

# ── ROI mode constant ──────────────────────────────────────────────────────────
ROI_FOREHEAD = "FOREHEAD"   # testa — 4 landmarks

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
FOREHEAD_ROI = [103, 332, 296, 66]


# ── ROI polygon ───────────────────────────────────────────────────────────────

def get_roi_polygon(landmarks, w, h, ema=None, alpha=0.35):
    """
    Returns the forehead ROI polygon with EMA smoothing.

    Parameters
    ----------
    landmarks : mediapipe NormalizedLandmarkList
    w, h      : int    Frame dimensions in pixels.
    ema       : np.ndarray or None   Previous EMA state.
    alpha     : float  EMA smoothing factor.

    Returns
    -------
    poly : np.ndarray, shape (4, 2), dtype int32
    ema  : np.ndarray  Updated EMA state.
    """
    points = np.array(
        [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
         for i in FOREHEAD_ROI], dtype=np.float32
    )
    if ema is None:
        ema = points.copy()
    else:
        ema = alpha * points + (1 - alpha) * ema
    return ema.astype(np.int32), ema
