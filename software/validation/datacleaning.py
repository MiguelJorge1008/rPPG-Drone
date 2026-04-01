"""
datacleaning.py — Recomputes rate (FFT) and variability (find_peaks) metrics
from the raw signals stored in recording CSVs, then overwrites them.

For each recording:
  - Rate  (HR / RR)         → FFT on sliding window, fs estimated from timestamps
  - Variability (SDNN/RMSSD or BRV) → find_peaks with distance based on fs

Both hardware (signal_hw) and camera (signal_sw) signals are processed.

Usage:
    python datacleaning.py                          # process all recordings
    python datacleaning.py --file Recording_X.csv   # process one file
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RECORDINGS  = os.path.join(SCRIPT_DIR, "recordings")

# Physiological bounds
HR_LOW_HZ  = 0.75   # 45 BPM
HR_HIGH_HZ = 3.33   # 200 BPM
RR_LOW_HZ  = 0.10   # 6 RPM
RR_HIGH_HZ = 0.70   # 42 RPM

# Sliding window sizes (seconds)
HR_WINDOW_S  = 20
RR_WINDOW_S  = 30
HRV_WINDOW_S = 30
BRV_WINDOW_S = 40

# Recompute every N rows (1 s at ~50 fps keeps resolution reasonable)
STEP_ROWS = 50


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def estimate_fs(timestamps):
    """Estimate sampling frequency from timestamps."""
    dt = np.median(np.diff(timestamps))
    return 1.0 / dt if dt > 0 else None


def preprocess(signal, fs, f_low, f_high, smooth_sigma=0.0):
    """
    Detrend + optional Gaussian smooth + bandpass.
    smooth_sigma > 0 helps when the signal is a step function (signal_sw rPPG).
    Returns filtered signal or None on failure.
    """
    sig = np.asarray(signal, dtype=float)
    if not np.isfinite(sig).any():
        return None

    # Replace NaNs with linear interpolation
    nans = ~np.isfinite(sig)
    if nans.any():
        idx = np.arange(len(sig))
        sig[nans] = np.interp(idx[nans], idx[~nans], sig[~nans])

    # Polynomial detrend (order 3 handles baseline drift)
    t = np.arange(len(sig))
    poly = np.polyfit(t, sig, 3)
    sig  = sig - np.polyval(poly, t)

    # Optional smoothing before bandpass (for step-function signals)
    if smooth_sigma > 0:
        sig = gaussian_filter1d(sig, sigma=smooth_sigma)

    # Bandpass
    nyq = fs / 2.0
    lo  = max(f_low  / nyq, 1e-4)
    hi  = min(f_high / nyq, 0.999)
    if lo >= hi:
        return None
    try:
        b, a = butter(4, [lo, hi], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rate via FFT
# ---------------------------------------------------------------------------

def fft_rate(signal_filt, fs, f_low, f_high):
    """
    Dominant frequency → rate in BPM/RPM.
    Includes sub-harmonic check: if dominant freq is 2×f and f/2 has a
    meaningful peak, use f/2 (avoids dicrotic notch doubling in HR).
    Returns float or np.nan.
    """
    n      = len(signal_filt)
    window = np.hanning(n)
    fft    = np.abs(np.fft.rfft(signal_filt * window))
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= f_low) & (freqs <= f_high)
    if not mask.any():
        return np.nan

    fft_band  = fft[mask]
    freqs_band = freqs[mask]
    peak_idx  = np.argmax(fft_band)
    f_dom     = freqs_band[peak_idx]

    # Sub-harmonic check: if f_dom/2 is in physiological range and its
    # FFT amplitude is at least 20% of f_dom, prefer f_dom/2
    f_half = f_dom / 2.0
    if f_low <= f_half <= f_high:
        half_mask = np.argmin(np.abs(freqs_band - f_half))
        if fft_band[half_mask] >= 0.20 * fft_band[peak_idx]:
            f_dom = f_half

    return f_dom * 60.0


def sliding_rate(timestamps, signal, f_low, f_high, window_s,
                 smooth_sigma=0.0, step=STEP_ROWS):
    """
    Compute time-varying rate using a backward-looking FFT window.
    Recomputes every `step` rows; intermediate rows are forward-filled.
    Returns array of length len(timestamps).
    """
    n   = len(timestamps)
    out = np.full(n, np.nan)
    fs  = estimate_fs(timestamps)
    if fs is None:
        return out

    w           = int(window_s * fs)
    min_samples = int(10 * fs)       # need at least 10 s
    last_val    = np.nan

    for i in range(n):
        if i % step == 0:
            start  = max(0, i - w)
            seg_t  = timestamps[start:i + 1]
            seg_s  = signal[start:i + 1]
            fs_seg = estimate_fs(seg_t)
            if fs_seg and len(seg_s) >= min_samples:
                filt = preprocess(seg_s, fs_seg, f_low, f_high,
                                  smooth_sigma=smooth_sigma)
                if filt is not None:
                    val = fft_rate(filt, fs_seg, f_low, f_high)
                    if np.isfinite(val):
                        last_val = val
        out[i] = last_val

    return out


# ---------------------------------------------------------------------------
# Variability via find_peaks
# ---------------------------------------------------------------------------

def sliding_hrv(timestamps, signal, window_s, step=STEP_ROWS):
    """
    Compute time-varying SDNN and RMSSD from beat-to-beat intervals.
    Peak distance is set to 400 ms (physiological minimum ~ 40 BPM).
    Returns (sdnn_arr, rmssd_arr), each of length len(timestamps).
    """
    n      = len(timestamps)
    sdnn   = np.full(n, np.nan)
    rmssd  = np.full(n, np.nan)
    fs     = estimate_fs(timestamps)
    if fs is None:
        return sdnn, rmssd

    w           = int(window_s * fs)
    min_samples = int(15 * fs)
    min_dist    = max(int(0.4 * fs), 1)   # 400 ms minimum between beats

    last_sdnn  = np.nan
    last_rmssd = np.nan

    for i in range(n):
        if i % step == 0:
            start = max(0, i - w)
            seg_t = timestamps[start:i + 1]
            seg_s = signal[start:i + 1]

            if len(seg_s) >= min_samples:
                filt = preprocess(seg_s, fs, HR_LOW_HZ, HR_HIGH_HZ)
                if filt is not None:
                    peaks, _ = find_peaks(filt, distance=min_dist,
                                          prominence=0.3 * np.std(filt))
                    if len(peaks) >= 3:
                        rr = np.diff(seg_t[peaks])
                        rr = rr[(rr > 0.3) & (rr < 1.5)]
                        if len(rr) >= 2:
                            med = np.median(rr)
                            mad = np.median(np.abs(rr - med))
                            rr  = rr[np.abs(rr - med) < 3 * mad]
                        if len(rr) >= 2:
                            last_sdnn  = np.std(rr) * 1000
                            last_rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000

        sdnn[i]  = last_sdnn
        rmssd[i] = last_rmssd

    return sdnn, rmssd


def sliding_brv(timestamps, signal, window_s, step=STEP_ROWS):
    """
    Compute time-varying BRV (breath-to-breath variability in ms).
    Peak distance set to 1.5 s minimum (max 40 RPM).
    Returns brv_arr of length len(timestamps).
    """
    n   = len(timestamps)
    out = np.full(n, np.nan)
    fs  = estimate_fs(timestamps)
    if fs is None:
        return out

    w           = int(window_s * fs)
    min_samples = int(20 * fs)
    min_dist    = max(int(1.5 * fs), 1)   # 1.5 s minimum between breaths

    last_val = np.nan

    for i in range(n):
        if i % step == 0:
            start = max(0, i - w)
            seg_t = timestamps[start:i + 1]
            seg_s = signal[start:i + 1]

            if len(seg_s) >= min_samples:
                filt = preprocess(seg_s, fs, RR_LOW_HZ, RR_HIGH_HZ)
                if filt is not None:
                    peaks, _ = find_peaks(filt, distance=min_dist,
                                          prominence=0.25 * np.std(filt))
                    if len(peaks) >= 3:
                        bb = np.diff(seg_t[peaks])
                        bb = bb[bb > 1.0]
                        if len(bb) >= 2:
                            last_val = np.std(bb) * 1000

        out[i] = last_val

    return out


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_rppg(df, fname):
    t  = df['timestamp'].values.astype(float)
    hw = df['signal_hw'].values.astype(float)
    sw = df['signal_sw'].values.astype(float)

    fs = estimate_fs(t)
    print(f"  fs={fs:.1f} Hz  n={len(df)}  duration={(t[-1]-t[0]):.0f}s")

    algo_cols = [c for c in df.columns if c.startswith("HR_") and c != "HR_gt"]
    algo = algo_cols[0].replace("HR_", "") if algo_cols else None

    # --- Hardware: FFT rate + HRV ---
    print("  [HW] Computing HR_gt via FFT ...")
    df['HR_gt'] = sliding_rate(t, hw, HR_LOW_HZ, HR_HIGH_HZ, HR_WINDOW_S)

    print("  [HW] Computing SDNN_gt / RMSSD_gt via find_peaks ...")
    sdnn_gt, rmssd_gt = sliding_hrv(t, hw, HRV_WINDOW_S)
    df['SDNN_gt']  = sdnn_gt
    df['RMSSD_gt'] = rmssd_gt

    # --- Camera: FFT rate + HRV ---
    # signal_sw is a step function (last BVP value per frame, updated every
    # ~30 frames). Gaussian smoothing (sigma=3) reduces step artifacts before FFT.
    if algo:
        print(f"  [SW] Computing HR_{algo} via FFT (smoothed signal_sw) ...")
        df[f'HR_{algo}'] = sliding_rate(t, sw, HR_LOW_HZ, HR_HIGH_HZ,
                                        HR_WINDOW_S, smooth_sigma=3.0)

        print(f"  [SW] Computing SDNN_{algo} / RMSSD_{algo} via find_peaks ...")
        sdnn_sw, rmssd_sw = sliding_hrv(t, sw, HRV_WINDOW_S)
        df[f'SDNN_{algo}']  = sdnn_sw
        df[f'RMSSD_{algo}'] = rmssd_sw

    return df


def process_resp(df, fname):
    t  = df['timestamp'].values.astype(float)
    hw = df['signal_hw'].values.astype(float)
    sw = df['signal_sw'].values.astype(float)

    fs = estimate_fs(t)
    print(f"  fs={fs:.1f} Hz  n={len(df)}  duration={(t[-1]-t[0]):.0f}s")

    # --- Hardware ---
    print("  [HW] Computing RR_gt via FFT ...")
    df['RR_gt'] = sliding_rate(t, hw, RR_LOW_HZ, RR_HIGH_HZ, RR_WINDOW_S)

    print("  [HW] Computing BRV_gt via find_peaks ...")
    df['BRV_gt'] = sliding_brv(t, hw, BRV_WINDOW_S)

    # --- Camera ---
    print("  [SW] Computing RR_BARTULA via FFT ...")
    df['RR_BARTULA'] = sliding_rate(t, sw, RR_LOW_HZ, RR_HIGH_HZ, RR_WINDOW_S)

    print("  [SW] Computing BRV_BARTULA via find_peaks ...")
    df['BRV_BARTULA'] = sliding_brv(t, sw, BRV_WINDOW_S)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recompute rate (FFT) and variability (find_peaks) from raw signals."
    )
    parser.add_argument('--file', type=str, default=None,
                        help="Process a single file (name only, inside recordings/)")
    args = parser.parse_args()

    if args.file:
        csv_files = [args.file]
    else:
        csv_files = sorted([f for f in os.listdir(RECORDINGS) if f.endswith('.csv')])

    if not csv_files:
        print("No CSV files found in recordings/")
        sys.exit(1)

    for fname in csv_files:
        fpath = os.path.join(RECORDINGS, fname)
        if not os.path.exists(fpath):
            print(f"[NOT FOUND] {fname}")
            continue

        print(f"\nProcessing: {fname}")
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        if 'signal_hw' not in df.columns or 'signal_sw' not in df.columns:
            print("  [SKIP] Missing signal_hw or signal_sw columns")
            continue

        is_rppg = 'HR_gt' in df.columns
        is_resp = 'RR_gt' in df.columns

        if is_rppg:
            df = process_rppg(df, fname)
        elif is_resp:
            df = process_resp(df, fname)
        else:
            print("  [SKIP] Unknown format")
            continue

        df.to_csv(fpath, index=False)
        print(f"  Saved: {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
