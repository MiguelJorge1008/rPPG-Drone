import cv2
import numpy as np

# ── ROI mode constants ─────────────────────────────────────────────────────────
ROI_FOREHEAD = "FOREHEAD"   # testa — 4 landmarks (original)
ROI_FACE     = "FACE"       # cara completa — 36 landmarks (face oval)
ROI_MULTI    = "MULTI"      # cara completa + grid 9×9 + DMRS (Face2PPG)

# ── MediaPipe landmark indices ─────────────────────────────────────────────────
FOREHEAD_ROI = [103, 332, 296, 66]

FACE_OVAL = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58,
             172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379,
             365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]

# ── Grid / DMRS parameters ────────────────────────────────────────────────────
GRID_N            = 9    # n×n grid cells across the face bounding box
R_MAX             = 32   # max regions kept after DMRS selection
MIN_REGION_PIXELS = 5    # min valid pixels for a cell to be used


# ── ROI polygon ───────────────────────────────────────────────────────────────

def get_roi_polygon(landmarks, w, h, ema=None, alpha=0.35, roi_mode=ROI_FACE):
    """
    Returns the ROI polygon for the given mode with EMA smoothing.

    Parameters
    ----------
    landmarks : mediapipe NormalizedLandmarkList
    w, h      : int    Frame dimensions in pixels.
    ema       : np.ndarray or None   Previous EMA state.
    alpha     : float  EMA smoothing factor.
    roi_mode  : str    ROI_FOREHEAD | ROI_FACE | ROI_MULTI

    Returns
    -------
    poly : np.ndarray, shape (N, 2), dtype int32
    ema  : np.ndarray  Updated EMA state.
    """
    indices = FOREHEAD_ROI if roi_mode == ROI_FOREHEAD else FACE_OVAL
    points  = np.array(
        [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
         for i in indices], dtype=np.float32
    )
    if ema is None:
        ema = points.copy()
    else:
        ema = alpha * points + (1 - alpha) * ema
    return ema.astype(np.int32), ema


# ── Grid RGB extraction ───────────────────────────────────────────────────────

def extract_grid_rgb(rgb, face_mask, poly, h, w, face_mean):
    """
    Divides the face bounding box into GRID_N × GRID_N cells and returns the
    mean RGB of each cell that has at least MIN_REGION_PIXELS inside the face mask.
    Cells with too few pixels fall back to face_mean.

    Parameters
    ----------
    rgb       : np.ndarray (h, w, 3)  RGB frame.
    face_mask : np.ndarray (h, w)     Binary mask of the face oval.
    poly      : np.ndarray (N, 2)     Face oval polygon (EMA-smoothed).
    h, w      : int   Frame dimensions.
    face_mean : np.ndarray (3,)       Mean RGB of the full face (fallback).

    Returns
    -------
    means : np.ndarray, shape (GRID_N*GRID_N, 3)
    valid : np.ndarray, shape (GRID_N*GRID_N,), dtype bool
        True for cells with enough skin pixels to be used in DMRS.
    """
    x_min = int(max(0, poly[:, 0].min()))
    x_max = int(min(w, poly[:, 0].max()))
    y_min = int(max(0, poly[:, 1].min()))
    y_max = int(min(h, poly[:, 1].max()))
    bw    = max(x_max - x_min, 1)
    bh    = max(y_max - y_min, 1)

    K     = GRID_N * GRID_N
    means = np.tile(face_mean.astype(np.float32), (K, 1))
    valid = np.zeros(K, dtype=bool)

    for row in range(GRID_N):
        for col in range(GRID_N):
            x0 = x_min + int(col       * bw / GRID_N)
            x1 = x_min + int((col + 1) * bw / GRID_N)
            y0 = y_min + int(row       * bh / GRID_N)
            y1 = y_min + int((row + 1) * bh / GRID_N)
            cell_face = face_mask[y0:y1, x0:x1]
            pixels    = rgb[y0:y1, x0:x1][cell_face == 255]
            if len(pixels) >= MIN_REGION_PIXELS:
                means[row * GRID_N + col] = pixels.mean(axis=0)
                valid[row * GRID_N + col] = True

    return means, valid


# ── DMRS — Dynamic Multi-Region Selection (Face2PPG) ─────────────────────────

def compute_kfd(signal):
    """
    Katz Fractal Dimension.
    D = log10(L/a) / log10(d/a)
    L = total path length, a = mean step, d = max dist from first point.
    """
    n = len(signal)
    if n < 3:
        return 0.0
    s     = signal.astype(np.float64)
    steps = np.sqrt(1.0 + np.diff(s) ** 2)
    L     = np.sum(steps)
    a     = L / (n - 1)
    d     = np.max(np.sqrt(np.arange(1, n) ** 2 + (s[1:] - s[0]) ** 2))
    if a == 0 or d == 0 or L == 0:
        return 0.0
    return np.log10(L / a) / np.log10(d / a)


def compute_dfa(signal, min_box=4):
    """
    Detrended Fluctuation Analysis — returns scaling exponent alpha.

    alpha ~0.5        uncorrelated noise
    0.5 < alpha < 1   positive long-range correlations (good rPPG)
    alpha < 0.5       anti-correlated (discard)
    """
    n = len(signal)
    if n < 16:
        return 0.5
    y      = np.cumsum(signal - np.mean(signal))
    max_box = n // 4
    scales  = np.unique(
        np.round(np.logspace(np.log10(min_box), np.log10(max_box), 20)).astype(int)
    )
    scales = scales[(scales >= min_box) & (scales <= max_box)]
    F, valid_scales = [], []
    for s in scales:
        n_boxes = n // s
        if n_boxes < 2:
            continue
        x = np.arange(s, dtype=np.float64)
        rms_list = [
            np.sqrt(np.mean(
                (y[k * s:(k + 1) * s] - np.polyval(np.polyfit(x, y[k * s:(k + 1) * s], 1), x)) ** 2
            ))
            for k in range(n_boxes)
        ]
        F.append(np.mean(rms_list))
        valid_scales.append(s)
    if len(F) < 2:
        return 0.5
    return float(np.polyfit(np.log10(valid_scales),
                            np.log10(np.maximum(F, 1e-12)), 1)[0])


def draw_grid_overlay(frame, poly, h, w, selected_indices):
    """
    Draws the GRID_N×GRID_N grid over the face bounding box.
    Selected regions are filled with a green semitransparent overlay.
    All cells get a thin cyan border.

    Parameters
    ----------
    frame           : np.ndarray (h, w, 3) BGR frame (modified in-place).
    poly            : np.ndarray (N, 2)    Face oval polygon.
    h, w            : int  Frame dimensions.
    selected_indices: list[int]  Indices of DMRS-selected regions.

    Returns
    -------
    np.ndarray  Annotated frame.
    """
    x_min = int(max(0, poly[:, 0].min()))
    x_max = int(min(w, poly[:, 0].max()))
    y_min = int(max(0, poly[:, 1].min()))
    y_max = int(min(h, poly[:, 1].max()))
    bw = max(x_max - x_min, 1)
    bh = max(y_max - y_min, 1)

    selected_set = set(selected_indices)
    overlay = frame.copy()

    for row in range(GRID_N):
        for col in range(GRID_N):
            idx = row * GRID_N + col
            x0 = x_min + int(col       * bw / GRID_N)
            x1 = x_min + int((col + 1) * bw / GRID_N)
            y0 = y_min + int(row       * bh / GRID_N)
            y1 = y_min + int((row + 1) * bh / GRID_N)
            if idx in selected_set:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 200), 1)

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    return frame


def dmrs_select(region_signals, fs, r_max=R_MAX, kfd_thresh=0.85, valid_mask=None):
    """
    Dynamic Multi-Region Selection (Face2PPG, Casado & Lopez 2023).
    DFA omitted for real-time performance — validity mask + KFD + spectral SNR.

    Parameters
    ----------
    region_signals : np.ndarray (N_frames, K_regions)
        Green-channel signal per region over time.
    fs         : float  Sampling frequency in Hz.
    r_max      : int    Max regions to return.
    kfd_thresh : float  Relative KFD threshold.
    valid_mask : np.ndarray (K_regions,) bool or None
        If provided, only regions marked True are considered.
        Cells outside the face oval (filled with face_mean fallback) must be
        excluded here to avoid contaminating the ranking.

    Returns
    -------
    list[int]  Indices of the best regions (up to r_max).
    """
    N, K = region_signals.shape

    # 0 — Restrict to cells with enough skin pixels (exclude face_mean fallbacks)
    candidates = [i for i in range(K) if valid_mask is None or valid_mask[i]]
    if not candidates:
        candidates = list(range(K))

    # 1 — Discard zero-variance regions
    valid = [i for i in candidates if np.var(region_signals[:, i]) > 0]
    if not valid:
        return list(range(min(r_max, K)))

    # 2 — KFD filter: keep regions with KFD >= kfd_thresh * global KFD
    global_kfd = compute_kfd(np.mean(region_signals[:, valid], axis=1))
    kfd_valid  = [i for i in valid
                  if global_kfd > 0 and
                  compute_kfd(region_signals[:, i]) / global_kfd >= kfd_thresh]
    if not kfd_valid:
        kfd_valid = valid

    # 3 — Rank by spectral SNR in HR band [0.75–4.0 Hz]
    #     SNR = band_energy / out-of-band_energy
    #     Noisy regions (beard, sides) have broadband energy → low SNR.
    #     Regions with a clean cardiac signal concentrate energy in the band → high SNR.
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    band  = (freqs >= 0.75) & (freqs <= 4.0)
    ffts  = np.abs(np.fft.rfft(region_signals[:, kfd_valid], axis=0)) ** 2
    snrs  = []
    for j in range(len(kfd_valid)):
        band_e  = float(np.sum(ffts[band, j]))
        total_e = float(np.sum(ffts[:, j]))
        snr     = band_e / (total_e - band_e + 1e-10)
        snrs.append((kfd_valid[j], snr))
    snrs.sort(key=lambda x: -x[1])
    return [i for i, _ in snrs[:r_max]]
