"""
ComputeMetrics.py — Post-process BVP and RESP recordings into metrics CSVs.

rPPG  : BVP_<ALGO>_<ts>_sw.csv + _hw.csv  ->  HR, SDNN, RMSSD
Resp  : RESP_<ts>_sw.csv       + _hw.csv  ->  RR, BBI

SW and HW have different sample rates — each is handled on its own time axis.

Usage:
    python ComputeMetrics.py                          # all pairs in data/
    python ComputeMetrics.py --file BVP_GREEN_X_sw.csv  # single pair
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(current_dir, "data")

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def metrics_from_peaks(sig, fs, min_dist_s=0.5):
    """HR (BPM), SDNN and RMSSD (ms) from peak intervals detected in sig."""
    if len(sig) < int(2 * fs):
        return np.nan, np.nan, np.nan
    peaks, _ = find_peaks(sig,
                          distance=max(1, int(min_dist_s * fs)),
                          prominence=np.std(sig) * 0.25)
    if len(peaks) < 3:
        return np.nan, np.nan, np.nan
    rri = np.diff(peaks) / fs          # seconds
    rri = rri[(rri > 0.4) & (rri < 1.5)]
    if len(rri) < 2:
        return np.nan, np.nan, np.nan
    hr    = 60.0 / np.mean(rri)
    sdnn  = np.std(rri) * 1000
    rmssd = np.sqrt(np.mean(np.diff(rri) ** 2)) * 1000
    return hr, sdnn, rmssd


def resp_metrics_from_peaks(sig, fs, min_dist_s=1.5):
    """RR (RPM) and mean BBI (s) from peak intervals in a respiratory signal."""
    if len(sig) < int(min_dist_s * 2 * fs):
        return np.nan, np.nan
    peaks, _ = find_peaks(sig,
                          distance=max(1, int(min_dist_s * fs)),
                          prominence=np.std(sig) * 0.25)
    if len(peaks) < 3:
        return np.nan, np.nan
    bbi = np.diff(peaks) / fs          # seconds
    bbi = bbi[(bbi >= 1.5) & (bbi <= 10.0)]
    if len(bbi) < 2:
        return np.nan, np.nan
    rr = 60.0 / np.mean(bbi)
    return rr, float(np.mean(bbi))


def bandpass(sig, fs, lo, hi, order=2):
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 0.001)
    hi_n = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype='bandpass')
    return filtfilt(b, a, sig.astype(float))


# ---------------------------------------------------------------------------
# Process one pair
# ---------------------------------------------------------------------------

def process_pair(sw_fname):
    """Compute metrics for one _sw / _hw pair. Returns output DataFrame or None."""
    # derive partner filename and algo
    base  = sw_fname.replace("_sw.csv", "")   # BVP_<ALGO>_<ts>
    parts = base.split("_")                    # ['BVP', algo, ts]
    if len(parts) < 3:
        print(f"  [SKIP] Cannot parse algo from {sw_fname}")
        return None, None
    algo  = parts[1]
    ts    = parts[2]
    hw_fname = f"{base}_hw.csv"

    sw_path = os.path.join(DATA_DIR, sw_fname)
    hw_path = os.path.join(DATA_DIR, hw_fname)

    if not os.path.exists(sw_path):
        print(f"  [NOT FOUND] {sw_fname}")
        return None, None
    if not os.path.exists(hw_path):
        print(f"  [NOT FOUND] {hw_fname}")
        return None, None

    df_sw_orig = pd.read_csv(sw_path)
    df_hw      = pd.read_csv(hw_path)

    t_sw  = df_sw_orig["timestamp"].values.astype(float)
    sig_sw = df_sw_orig["signal_sw"].values.astype(float)

    t_hw   = df_hw["timestamp"].values.astype(float)
    sig_hw = df_hw["signal_hw"].values.astype(float)

    # effective sample rates
    fs_sw = 1.0 / np.median(np.diff(t_sw)) if len(t_sw) > 1 else 25.0
    fs_hw = 1.0 / np.median(np.diff(t_hw)) if len(t_hw) > 1 else 100.0

    hw_all_zero = np.all(sig_hw == 0)

    # interpolate HW signal to SW timestamps for the output signal_hw column
    if not hw_all_zero and len(t_hw) > 1:
        sig_hw_interp = np.interp(t_sw, t_hw, sig_hw)
    else:
        sig_hw_interp = np.zeros(len(t_sw))

    # deduplicate SW signal (remove consecutive identical values = plateaus)
    dd_idx    = np.concatenate([[True], np.diff(sig_sw) != 0])
    t_sw_dd   = t_sw[dd_idx]
    sig_sw_dd = sig_sw[dd_idx]
    fs_sw_dd  = 1.0 / np.median(np.diff(t_sw_dd)) if len(t_sw_dd) > 1 else fs_sw

    print(f"  SW: {len(t_sw)} samples @ {fs_sw:.1f} Hz  "
          f"({len(t_sw_dd)} unique)  |  "
          f"HW: {len(t_hw)} samples @ {fs_hw:.1f} Hz")

    rows = []
    for i, t_i in enumerate(t_sw):
        # --- SW window: deduped data from start up to t_i ---
        sw_dd_mask = t_sw_dd <= t_i
        sw_win_dd  = sig_sw_dd[sw_dd_mask]

        hr_sw    = np.nan
        sdnn_sw  = np.nan
        rmssd_sw = np.nan

        if len(sw_win_dd) >= int(fs_sw_dd * 5):
            hr_sw, sdnn_sw, rmssd_sw = metrics_from_peaks(sw_win_dd, fs_sw_dd)

        # --- HW window: all data from start up to t_i ---
        hr_hw    = np.nan
        sdnn_hw  = np.nan
        rmssd_hw = np.nan

        if not hw_all_zero and len(t_hw) > 1:
            hw_mask = t_hw <= t_i
            hw_win  = sig_hw[hw_mask]

            if len(hw_win) >= int(fs_hw * 5):
                try:
                    hw_filt = bandpass(hw_win, fs_hw, 0.75, 4.0)
                except Exception:
                    hw_filt = hw_win
                hr_hw, sdnn_hw, rmssd_hw = metrics_from_peaks(hw_filt, fs_hw)

        rows.append({
            'HR_gt':         hr_hw,
            f'HR_{algo}':    hr_sw,
            'SDNN_gt':       sdnn_hw,
            f'SDNN_{algo}':  sdnn_sw,
            'RMSSD_gt':      rmssd_hw,
            f'RMSSD_{algo}': rmssd_sw,
        })

    df_metrics = pd.DataFrame(rows)

    # SW file: only SW metrics
    for col in [f'HR_{algo}', f'SDNN_{algo}', f'RMSSD_{algo}']:
        df_sw_orig[col] = df_metrics[col].values

    # HW file: only HW metrics (on HW timestamps)
    df_hw_out = None
    if not hw_all_zero:
        hw_rows = []
        for j, t_j in enumerate(t_hw):
            hw_mask = t_hw <= t_j
            hw_win  = sig_hw[hw_mask]
            hr_hw = sdnn_hw = rmssd_hw = np.nan
            if len(hw_win) >= int(fs_hw * 5):
                try:
                    hw_filt = bandpass(hw_win, fs_hw, 0.75, 4.0)
                except Exception:
                    hw_filt = hw_win
                hr_hw, sdnn_hw, rmssd_hw = metrics_from_peaks(hw_filt, fs_hw)
            hw_rows.append({'HR_gt': hr_hw, 'SDNN_gt': sdnn_hw, 'RMSSD_gt': rmssd_hw})

        df_hw_metrics = pd.DataFrame(hw_rows)
        df_hw_out = pd.read_csv(hw_path)
        for col in ['HR_gt', 'SDNN_gt', 'RMSSD_gt']:
            df_hw_out[col] = df_hw_metrics[col].values

    return df_sw_orig, df_hw_out, algo


# ---------------------------------------------------------------------------
# Process one RESP pair
# ---------------------------------------------------------------------------

def process_resp_pair(sw_fname):
    """Compute RR and BBI for one RESP _sw / _hw pair."""
    base     = sw_fname.replace("_sw.csv", "")   # RESP_<ts>
    hw_fname = f"{base}_hw.csv"

    sw_path = os.path.join(DATA_DIR, sw_fname)
    hw_path = os.path.join(DATA_DIR, hw_fname)

    if not os.path.exists(sw_path):
        print(f"  [NOT FOUND] {sw_fname}")
        return None, None
    if not os.path.exists(hw_path):
        print(f"  [NOT FOUND] {hw_fname}")
        return None, None

    df_sw_orig = pd.read_csv(sw_path)
    df_hw      = pd.read_csv(hw_path)

    t_sw   = df_sw_orig["timestamp"].values.astype(float)
    sig_sw = df_sw_orig["signal_sw"].values.astype(float)

    t_hw   = df_hw["timestamp"].values.astype(float)
    sig_hw = df_hw["signal_hw"].values.astype(float)

    fs_sw = 1.0 / np.median(np.diff(t_sw)) if len(t_sw) > 1 else 25.0
    fs_hw = 1.0 / np.median(np.diff(t_hw)) if len(t_hw) > 1 else 100.0

    hw_all_zero = np.all(sig_hw == 0)

    print(f"  SW: {len(t_sw)} samples @ {fs_sw:.1f} Hz  |  "
          f"HW: {len(t_hw)} samples @ {fs_hw:.1f} Hz")

    # --- SW growing window ---
    rows_sw = []
    for t_i in t_sw:
        sw_mask = t_sw <= t_i
        sw_win  = sig_sw[sw_mask]
        rr_sw = bbi_sw = np.nan
        if len(sw_win) >= int(fs_sw * 10):   # need ~2 breaths at min 5 RPM
            rr_sw, bbi_sw = resp_metrics_from_peaks(sw_win, fs_sw)
        rows_sw.append({'RR_BARTULA': rr_sw, 'BBI_BARTULA': bbi_sw})

    df_sw_metrics = pd.DataFrame(rows_sw)
    for col in ['RR_BARTULA', 'BBI_BARTULA']:
        df_sw_orig[col] = df_sw_metrics[col].values

    # --- HW growing window (own time axis) ---
    df_hw_out = None
    if not hw_all_zero:
        rows_hw = []
        for t_j in t_hw:
            hw_mask = t_hw <= t_j
            hw_win  = sig_hw[hw_mask]
            rr_hw = bbi_hw = np.nan
            if len(hw_win) >= int(fs_hw * 10):
                rr_hw, bbi_hw = resp_metrics_from_peaks(hw_win, fs_hw)
            rows_hw.append({'RR_gt': rr_hw, 'BBI_gt': bbi_hw})

        df_hw_metrics = pd.DataFrame(rows_hw)
        df_hw_out = pd.read_csv(hw_path)
        for col in ['RR_gt', 'BBI_gt']:
            df_hw_out[col] = df_hw_metrics[col].values

    return df_sw_orig, df_hw_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Single _sw.csv file to process (name only)")
    args = parser.parse_args()

    all_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.endswith("_sw.csv")
    )

    if args.file:
        all_files = [args.file]

    bvp_files  = [f for f in all_files if f.startswith("BVP_")]
    resp_files = [f for f in all_files if f.startswith("RESP_")]

    if not bvp_files and not resp_files:
        print("No BVP_ or RESP_ _sw.csv files found in data/")
        sys.exit(1)

    for sw_fname in bvp_files:
        print(f"\nProcessing rPPG: {sw_fname}")
        df_sw_out, df_hw_out, algo = process_pair(sw_fname)
        if df_sw_out is None:
            continue
        sw_path = os.path.join(DATA_DIR, sw_fname)
        df_sw_out.to_csv(sw_path, index=False)
        print(f"  Updated SW: {sw_fname}  ({len(df_sw_out)} rows)")
        if df_hw_out is not None:
            hw_fname = sw_fname.replace("_sw.csv", "_hw.csv")
            df_hw_out.to_csv(os.path.join(DATA_DIR, hw_fname), index=False)
            print(f"  Updated HW: {hw_fname}  ({len(df_hw_out)} rows)")

    for sw_fname in resp_files:
        print(f"\nProcessing Resp: {sw_fname}")
        df_sw_out, df_hw_out = process_resp_pair(sw_fname)
        if df_sw_out is None:
            continue
        sw_path = os.path.join(DATA_DIR, sw_fname)
        df_sw_out.to_csv(sw_path, index=False)
        print(f"  Updated SW: {sw_fname}  ({len(df_sw_out)} rows)")
        if df_hw_out is not None:
            hw_fname = sw_fname.replace("_sw.csv", "_hw.csv")
            df_hw_out.to_csv(os.path.join(DATA_DIR, hw_fname), index=False)
            print(f"  Updated HW: {hw_fname}  ({len(df_hw_out)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
