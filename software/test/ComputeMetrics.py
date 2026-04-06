"""
ComputeMetrics.py — Post-process BVP and RESP recordings into metrics.

Outputs to data_processed/:
  BVP_<ALGO>_<ts>_sw_processed.csv  — timestamp, signal_sw, HR_<algo>, SDNN_<algo>, RMSSD_<algo>
  BVP_<ALGO>_<ts>_hw_processed.csv  — timestamp, signal_hw, HR_gt,    SDNN_gt,    RMSSD_gt
  RESP_<ts>_sw_processed.csv        — timestamp, signal_sw, RR_BARTULA, BBI_BARTULA
  RESP_<ts>_hw_processed.csv        — timestamp, signal_hw, RR_gt,      BBI_gt

Usage:
    python ComputeMetrics.py                             # all pairs in data/
    python ComputeMetrics.py --file BVP_GREEN_X_sw.csv  # single pair
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

current_dir       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(current_dir, "data")
DATA_PROCESSED_DIR = os.path.join(current_dir, "data_processed")

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
    rri = np.diff(peaks) / fs
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
    bbi = np.diff(peaks) / fs
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
# Process one rPPG pair
# ---------------------------------------------------------------------------

def process_pair(sw_fname):
    """Returns (df_sw_out, df_hw_out, algo) or (None, None, None)."""
    m = re.match(r'^(.+?)(_sw)((?:_[A-Za-z]+)*)\.csv$', sw_fname)
    if not m:
        print(f"  [SKIP] Cannot parse filename {sw_fname}")
        return None, None, None
    base, _, sw_extra = m.group(1), m.group(2), m.group(3)
    parts = base.split("_")
    # timestamp is the last all-digit part; algo is everything between BVP_ and it
    try:
        ts_idx = next(i for i, p in enumerate(parts) if p.isdigit())
    except StopIteration:
        print(f"  [SKIP] Cannot parse algo from {sw_fname}")
        return None, None, None
    algo     = "_".join(parts[1:ts_idx])
    hw_fname = f"{base}_hw{sw_extra}.csv"

    sw_path = os.path.join(DATA_DIR, sw_fname)
    hw_path = os.path.join(DATA_DIR, hw_fname)

    if not os.path.exists(sw_path):
        print(f"  [NOT FOUND] {sw_fname}")
        return None, None, None
    if not os.path.exists(hw_path):
        print(f"  [NOT FOUND] {hw_fname}")
        return None, None, None

    df_sw = pd.read_csv(sw_path)
    df_hw = pd.read_csv(hw_path)

    t_sw   = df_sw["timestamp"].values.astype(float)
    sig_sw = df_sw["signal_sw"].values.astype(float)
    t_hw   = df_hw["timestamp"].values.astype(float)
    sig_hw = df_hw["signal_hw"].values.astype(float)

    fs_sw = 1.0 / np.median(np.diff(t_sw)) if len(t_sw) > 1 else 25.0
    fs_hw = 1.0 / np.median(np.diff(t_hw)) if len(t_hw) > 1 else 100.0

    hw_all_zero = np.all(sig_hw == 0)

    dd_idx    = np.concatenate([[True], np.diff(sig_sw) != 0])
    t_sw_dd   = t_sw[dd_idx]
    sig_sw_dd = sig_sw[dd_idx]
    fs_sw_dd  = 1.0 / np.median(np.diff(t_sw_dd)) if len(t_sw_dd) > 1 else fs_sw

    print(f"  SW: {len(t_sw)} samples @ {fs_sw:.1f} Hz  "
          f"({len(t_sw_dd)} unique)  |  "
          f"HW: {len(t_hw)} samples @ {fs_hw:.1f} Hz")

    # --- SW loop: metrics per SW timestamp ---
    sw_rows = []
    for i, t_i in enumerate(t_sw):
        sw_dd_mask = t_sw_dd <= t_i
        sw_win_dd  = sig_sw_dd[sw_dd_mask]
        hr_sw = sdnn_sw = rmssd_sw = np.nan
        if len(sw_win_dd) >= int(fs_sw_dd * 5):
            hr_sw, sdnn_sw, rmssd_sw = metrics_from_peaks(sw_win_dd, fs_sw_dd)
        sw_rows.append({
            'timestamp':      t_i,
            'signal_sw':      sig_sw[i],
            f'HR_{algo}':     hr_sw,
            f'SDNN_{algo}':   sdnn_sw,
            f'RMSSD_{algo}':  rmssd_sw,
        })

    # --- HW loop: metrics per HW timestamp ---
    hw_rows = []
    if not hw_all_zero:
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
            hw_rows.append({
                'timestamp': t_j,
                'signal_hw': sig_hw[j],
                'HR_gt':     hr_hw,
                'SDNN_gt':   sdnn_hw,
                'RMSSD_gt':  rmssd_hw,
            })

    df_sw_out = pd.DataFrame(sw_rows)
    df_hw_out = pd.DataFrame(hw_rows) if hw_rows else None
    return df_sw_out, df_hw_out, algo


# ---------------------------------------------------------------------------
# Process one RESP pair
# ---------------------------------------------------------------------------

def process_resp_pair(sw_fname):
    """Returns (df_sw_out, df_hw_out) or (None, None)."""
    base     = sw_fname.replace("_sw.csv", "")
    hw_fname = f"{base}_hw.csv"

    sw_path = os.path.join(DATA_DIR, sw_fname)
    hw_path = os.path.join(DATA_DIR, hw_fname)

    if not os.path.exists(sw_path):
        print(f"  [NOT FOUND] {sw_fname}")
        return None, None
    if not os.path.exists(hw_path):
        print(f"  [NOT FOUND] {hw_fname}")
        return None, None

    df_sw = pd.read_csv(sw_path)
    df_hw = pd.read_csv(hw_path)

    t_sw   = df_sw["timestamp"].values.astype(float)
    sig_sw = df_sw["signal_sw"].values.astype(float)
    t_hw   = df_hw["timestamp"].values.astype(float)
    sig_hw = df_hw["signal_hw"].values.astype(float)

    fs_sw = 1.0 / np.median(np.diff(t_sw)) if len(t_sw) > 1 else 25.0
    fs_hw = 1.0 / np.median(np.diff(t_hw)) if len(t_hw) > 1 else 100.0

    hw_all_zero = np.all(sig_hw == 0)

    # downsample SW to at most 25 Hz — respiratory signals need no more
    TARGET_FS = 25.0
    if fs_sw > TARGET_FS * 1.1:
        step   = max(1, int(round(fs_sw / TARGET_FS)))
        t_sw   = t_sw[::step]
        sig_sw = sig_sw[::step]
        fs_sw  = 1.0 / np.median(np.diff(t_sw))

    print(f"  SW: {len(t_sw)} samples @ {fs_sw:.1f} Hz  |  "
          f"HW: {len(t_hw)} samples @ {fs_hw:.1f} Hz")

    SLIDE_S = 30.0

    # --- SW loop ---
    sw_rows = []
    for idx, t_i in enumerate(t_sw):
        sw_mask = (t_sw <= t_i) & (t_sw > t_i - SLIDE_S)
        sw_win  = sig_sw[sw_mask]
        rr_sw = bbi_sw = np.nan
        if len(sw_win) >= int(fs_sw * SLIDE_S * 0.5):
            rr_sw, bbi_sw = resp_metrics_from_peaks(sw_win, fs_sw)
        sw_rows.append({
            'timestamp':   t_i,
            'signal_sw':   sig_sw[idx],
            'RR_BARTULA':  rr_sw,
            'BBI_BARTULA': bbi_sw,
        })

    # --- HW loop ---
    hw_rows = []
    if not hw_all_zero:
        for j, t_j in enumerate(t_hw):
            hw_mask = (t_hw <= t_j) & (t_hw > t_j - SLIDE_S)
            hw_win  = sig_hw[hw_mask]
            rr_hw = bbi_hw = np.nan
            if len(hw_win) >= int(fs_hw * SLIDE_S * 0.5):
                rr_hw, bbi_hw = resp_metrics_from_peaks(hw_win, fs_hw)
            hw_rows.append({
                'timestamp': t_j,
                'signal_hw': sig_hw[j],
                'RR_gt':     rr_hw,
                'BBI_gt':    bbi_hw,
            })

    df_sw_out = pd.DataFrame(sw_rows)
    df_hw_out = pd.DataFrame(hw_rows) if hw_rows else None
    return df_sw_out, df_hw_out


# ---------------------------------------------------------------------------
# Write helper (create or update existing processed file)
# ---------------------------------------------------------------------------

def _write_processed(df_new, path, skip_cols):
    """Write df_new to path. If file exists, update metric columns only."""
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        for col in df_new.columns:
            if col not in skip_cols:
                df_existing[col] = df_new[col].values
        df_new = df_existing
    df_new.to_csv(path, index=False)
    print(f"  -> {os.path.basename(path)}  ({len(df_new)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Single _sw.csv file to process (name only)")
    args = parser.parse_args()

    all_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if re.search(r'_sw(?:_[A-Za-z]+)?\.csv$', f)
    )
    if args.file:
        all_files = [args.file]

    bvp_files  = [f for f in all_files if f.startswith("BVP_")]
    resp_files = [f for f in all_files if f.startswith("RESP_")]

    if not bvp_files and not resp_files:
        print("No BVP_ or RESP_ _sw.csv files found in data/")
        sys.exit(1)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    for sw_fname in bvp_files:
        print(f"\nProcessing rPPG: {sw_fname}")
        df_sw_out, df_hw_out, algo = process_pair(sw_fname)
        if df_sw_out is None:
            continue
        m = re.match(r'^(.+?)(_sw)((?:_[A-Za-z]+)*)\.csv$', sw_fname)
        base, sw_extra = m.group(1), m.group(3)
        _write_processed(df_sw_out,
                         os.path.join(DATA_PROCESSED_DIR, f"{base}_sw{sw_extra}_processed.csv"),
                         skip_cols=('timestamp', 'signal_sw'))
        if df_hw_out is not None:
            _write_processed(df_hw_out,
                             os.path.join(DATA_PROCESSED_DIR, f"{base}_hw{sw_extra}_processed.csv"),
                             skip_cols=('timestamp', 'signal_hw'))
        print(df_sw_out.tail(3).to_string(index=False))

    for sw_fname in resp_files:
        print(f"\nProcessing Resp: {sw_fname}")
        df_sw_out, df_hw_out = process_resp_pair(sw_fname)
        if df_sw_out is None:
            continue
        m = re.match(r'^(.+?)(_sw)((?:_[A-Za-z]+)*)\.csv$', sw_fname)
        base, sw_extra = m.group(1), m.group(3)
        _write_processed(df_sw_out,
                         os.path.join(DATA_PROCESSED_DIR, f"{base}_sw{sw_extra}_processed.csv"),
                         skip_cols=('timestamp', 'signal_sw'))
        if df_hw_out is not None:
            _write_processed(df_hw_out,
                             os.path.join(DATA_PROCESSED_DIR, f"{base}_hw{sw_extra}_processed.csv"),
                             skip_cols=('timestamp', 'signal_hw'))
        print(df_sw_out.tail(3).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
