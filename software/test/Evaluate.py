"""
evaluate.py — Evaluate rPPG / respiration recordings.

Reads HR_gt + HR_<algo>  (or RR_gt + RR_BARTULA) directly from each CSV
and computes:  MAE ± SD,  RMSE,  Bias,  PCC

Also evaluates variability metrics when present in the CSV:
  - rPPG:  SDNN_gt / SDNN_<algo>  and  RMSSD_gt / RMSSD_<algo>  (ms)
  - Resp:  BRV_gt  / BRV_BARTULA                                  (ms)

GT columns must already be present in the CSV — run reprocess_gt.py first
if the recordings were made before the GT computation was updated.

Usage:
    python evaluate.py                          # all recordings (raw)
    python evaluate.py --file Recording_X.csv   # single file
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _normalise(s):
    s = np.asarray(s, dtype=float)
    std = np.nanstd(s)
    return (s - np.nanmean(s)) / std if std > 0 else s - np.nanmean(s)


def _detect_peaks(sig, fs, min_dist_s):
    if len(sig) < int(2 * fs):
        return np.array([], dtype=int), sig
    peaks, _ = find_peaks(sig,
                          distance=max(1, int(min_dist_s * fs)),
                          prominence=np.std(sig) * 0.25)
    return peaks, sig


def _ibi(peak_idx, t):
    if len(peak_idx) < 2:
        return np.array([]), np.array([])
    return (t[peak_idx[:-1]] + t[peak_idx[1:]]) / 2.0, np.diff(t[peak_idx])


def _prep_signal(series, fs):
    return pd.Series(series).interpolate(limit=5).values


def _dedup(sig, t):
    sig = np.asarray(sig, dtype=float)
    if len(sig) == 0:
        return np.array([], dtype=int)
    return np.concatenate([[0], np.where(np.diff(sig) != 0)[0] + 1])


def _plot_ibi(ax, hw_mid, hw_ibi, sw_mid, sw_ibi,
              ylabel, valid_lo, valid_hi, hw_color, sw_color):
    if len(hw_ibi):
        ax.plot(hw_mid, hw_ibi, "o-", color=hw_color, ms=4, lw=1.0,
                label=f"HW  mean={np.mean(hw_ibi):.3f}{ylabel.split('(')[1].rstrip(')')}")
    if len(sw_ibi):
        ax.plot(sw_mid, sw_ibi, "^-", color=sw_color, ms=4, lw=1.0, alpha=0.8,
                label=f"SW  mean={np.mean(sw_ibi):.3f}{ylabel.split('(')[1].rstrip(')')}")
    ax.axhspan(valid_lo, valid_hi, color="gray", alpha=0.08,
               label=f"Valid [{valid_lo}–{valid_hi}]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def _save_plot(fig, fname, out_dir):
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, fname.replace(".csv", ".png"))
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


def plot_rppg(df_sw, fname, out_dir, df_hw=None):
    algo_cols = [c for c in df_sw.columns if c.startswith("HR_") and c != "HR_gt"]
    if not algo_cols:
        return
    algo_col = algo_cols[0]
    algo     = algo_col.replace("HR_", "")

    PLOT_DUR = 60.0

    t_sw = df_sw["timestamp"].values.astype(float)
    t0   = t_sw[0]
    t_sw = t_sw - t0
    sw_mask = t_sw <= PLOT_DUR
    t_sw    = t_sw[sw_mask]
    df_sw   = df_sw.iloc[sw_mask]
    fs_sw = 1.0 / np.median(np.diff(t_sw[np.isfinite(t_sw)])) if len(t_sw) > 1 else 25.0

    sw_full  = _prep_signal(df_sw["signal_sw"].values, fs_sw)
    sw_idx   = _dedup(sw_full, t_sw)
    sw       = sw_full[sw_idx]
    t_sw_dd  = t_sw[sw_idx]
    fs_sw_dd = 1.0 / np.median(np.diff(t_sw_dd)) if len(t_sw_dd) > 1 else fs_sw

    sw_peaks, sw_filt = _detect_peaks(sw, fs_sw_dd, min_dist_s=0.3)
    sw_filt_n = _normalise(sw_filt)
    sw_mid, sw_ibi = _ibi(sw_peaks, t_sw_dd)

    has_hw = (df_hw is not None and
              "signal_hw" in df_hw.columns and
              df_hw["signal_hw"].notna().any() and
              not np.all(df_hw["signal_hw"].values == 0))

    if has_hw:
        t_hw   = df_hw["timestamp"].values.astype(float) - t0
        hw_mask = (t_hw >= 0) & (t_hw <= PLOT_DUR)
        t_hw    = t_hw[hw_mask]
        sig_hw  = df_hw["signal_hw"].values.astype(float)[hw_mask]
        fs_hw  = 1.0 / np.median(np.diff(t_hw)) if len(t_hw) > 1 else 100.0
        hw     = _prep_signal(sig_hw, fs_hw)
        hw_peaks, hw_filt = _detect_peaks(hw, fs_hw, min_dist_s=0.3)
        hw_filt_n = _normalise(hw_filt)
        hw_mid, hw_ibi = _ibi(hw_peaks, t_hw)
    else:
        t_hw = np.array([])
        hw_peaks, hw_filt_n = np.array([], dtype=int), np.array([])
        hw_mid, hw_ibi = np.array([]), np.array([])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(fname, fontsize=11, fontweight="bold")

    ax = axes[0]
    if has_hw:
        ax.plot(t_hw,    hw_filt_n, color="red",   lw=0.8, alpha=0.85, label="HW PPG (z-norm)")
    ax.plot(t_sw_dd, sw_filt_n, color="green", lw=0.8, alpha=0.75, label="SW rPPG (z-norm)")
    if has_hw and len(hw_peaks):
        ax.scatter(t_hw[hw_peaks],    hw_filt_n[hw_peaks],    color="darkred",   s=28, zorder=5, label=f"HW peaks ({len(hw_peaks)})")
    if len(sw_peaks):
        ax.scatter(t_sw_dd[sw_peaks], sw_filt_n[sw_peaks],    color="darkgreen", s=28, zorder=5, marker="^", label=f"SW peaks ({len(sw_peaks)})")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title("Signals (z-score normalised) with detected peaks")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if has_hw and "HR_gt" in df_sw.columns:
        axes[1].plot(t_sw, df_sw["HR_gt"],  color="red",   lw=1.2, label="HR_gt  (hardware)")
    axes[1].plot(t_sw, df_sw[algo_col],     color="green", lw=1.2, label=f"{algo_col}  (camera)", alpha=0.8)
    axes[1].set_ylabel("BPM")
    axes[1].set_title("Heart Rate over time")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    sdnn_col  = f"SDNN_{algo}"
    rmssd_col = f"RMSSD_{algo}"
    has_hrv   = sdnn_col in df_sw.columns or rmssd_col in df_sw.columns
    ax2 = axes[2]
    if has_hrv:
        if "SDNN_gt" in df_sw.columns and has_hw:
            ax2.plot(t_sw, df_sw["SDNN_gt"],  color="red",       lw=1.0, linestyle="--", label="SDNN_gt")
        if sdnn_col in df_sw.columns:
            ax2.plot(t_sw, df_sw[sdnn_col],   color="green",     lw=1.0, label=f"SDNN_{algo}")
        if "RMSSD_gt" in df_sw.columns and has_hw:
            ax2.plot(t_sw, df_sw["RMSSD_gt"], color="darkred",   lw=1.0, linestyle="--", label="RMSSD_gt")
        if rmssd_col in df_sw.columns:
            ax2.plot(t_sw, df_sw[rmssd_col],  color="darkgreen", lw=1.0, linestyle="-.", label=f"RMSSD_{algo}")
        ax2.set_ylabel("HRV (ms)")
        ax2.set_title("HRV metrics over time (SDNN / RMSSD)")
    else:
        _plot_ibi(ax2, hw_mid, hw_ibi * 1000 if len(hw_ibi) else hw_ibi,
                  sw_mid, sw_ibi * 1000,
                  ylabel="IBI (ms)", valid_lo=300, valid_hi=1500,
                  hw_color="red", sw_color="green")
        ax2.set_title("Inter-Beat Intervals over time")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (s)")

    _save_plot(fig, fname, out_dir)


def plot_resp(df_sw, fname, out_dir, df_hw=None):
    PLOT_DUR = 60.0

    t_sw = df_sw["timestamp"].values.astype(float)
    t0   = t_sw[0]
    t_sw = t_sw - t0
    sw_mask = t_sw <= PLOT_DUR
    t_sw    = t_sw[sw_mask]
    df_sw   = df_sw.iloc[sw_mask]
    fs_sw = 1.0 / np.median(np.diff(t_sw[np.isfinite(t_sw)])) if len(t_sw) > 1 else 25.0

    sw_full  = _prep_signal(df_sw["signal_sw"].values, fs_sw)
    sw_idx   = _dedup(sw_full, t_sw)
    sw       = sw_full[sw_idx]
    t_sw_dd  = t_sw[sw_idx]
    fs_sw_dd = 1.0 / np.median(np.diff(t_sw_dd)) if len(t_sw_dd) > 1 else fs_sw

    sw_peaks, sw_filt = _detect_peaks(sw, fs_sw_dd, min_dist_s=1.5)
    sw_filt_n = _normalise(sw_filt)

    has_hw = (df_hw is not None and
              "signal_hw" in df_hw.columns and
              df_hw["signal_hw"].notna().any() and
              not np.all(df_hw["signal_hw"].values == 0))

    if has_hw:
        t_hw    = df_hw["timestamp"].values.astype(float) - t0
        hw_mask = (t_hw >= 0) & (t_hw <= PLOT_DUR)
        t_hw    = t_hw[hw_mask]
        sig_hw  = df_hw["signal_hw"].values.astype(float)[hw_mask]
        fs_hw  = 1.0 / np.median(np.diff(t_hw)) if len(t_hw) > 1 else 100.0
        hw     = _prep_signal(sig_hw, fs_hw)
        hw_peaks, hw_filt = _detect_peaks(hw, fs_hw, min_dist_s=1.5)
        hw_filt_n = _normalise(hw_filt)
        hw_mid, hw_bbi = _ibi(hw_peaks, t_hw)
    else:
        t_hw = np.array([])
        hw_peaks, hw_filt_n = np.array([], dtype=int), np.array([])
        hw_mid, hw_bbi = np.array([]), np.array([])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(fname, fontsize=11, fontweight="bold")

    ax = axes[0]
    if has_hw:
        ax.plot(t_hw,    hw_filt_n, color="blue", lw=0.8, alpha=0.85, label="HW resp (z-norm)")
    ax.plot(t_sw_dd, sw_filt_n, color="cyan", lw=0.8, alpha=0.75, label="SW resp (z-norm)")
    if has_hw and len(hw_peaks):
        ax.scatter(t_hw[hw_peaks],    hw_filt_n[hw_peaks],    color="darkblue", s=28, zorder=5, label=f"HW peaks ({len(hw_peaks)})")
    if len(sw_peaks):
        ax.scatter(t_sw_dd[sw_peaks], sw_filt_n[sw_peaks],    color="darkcyan", s=28, zorder=5, marker="^", label=f"SW peaks ({len(sw_peaks)})")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title("Signals (z-score normalised) with detected peaks")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    if has_hw and "RR_gt" in df_sw.columns:
        axes[1].plot(t_sw, df_sw["RR_gt"],     color="blue", lw=1.2, label="RR_gt  (hardware)")
    axes[1].plot(t_sw, df_sw["RR_BARTULA"],     color="cyan", lw=1.2, label="RR_BARTULA  (camera)", alpha=0.8)
    axes[1].set_ylabel("RPM")
    axes[1].set_title("Respiration Rate over time")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    ax = axes[2]
    if "BBI_BARTULA" in df_sw.columns:
        if has_hw and "BBI_gt" in df_sw.columns:
            ax.plot(t_sw, df_sw["BBI_gt"],     color="blue", lw=1.2, label="BBI_gt  (hardware)")
        ax.plot(t_sw, df_sw["BBI_BARTULA"],     color="cyan", lw=1.2, label="BBI_BARTULA  (camera)", alpha=0.8)
        ax.axhspan(1.5, 10.0, color="gray", alpha=0.08, label="Valid [1.5–10 s]")
        ax.set_ylabel("BBI (s)")
        ax.legend(fontsize=8, loc="upper right")
    else:
        sw_mid, sw_bbi = _ibi(sw_peaks, t_sw_dd)
        _plot_ibi(ax, hw_mid, hw_bbi, sw_mid, sw_bbi,
                  ylabel="BBI (s)", valid_lo=1.5, valid_hi=10.0,
                  hw_color="blue", sw_color="cyan")
    ax.set_title("Breath-to-Breath Intervals over time")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    _save_plot(fig, fname, out_dir)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RECORDINGS  = os.path.join(SCRIPT_DIR, "data_processed")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_OUT     = os.path.join(RESULTS_DIR, "csv_raw")
PLOTS_EVAL  = RESULTS_DIR


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt, est):
    gt  = np.asarray(gt,  dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(gt) & np.isfinite(est)
    gt, est = gt[mask], est[mask]
    if len(gt) < 3:
        return dict(n=len(gt), mae=np.nan, mae_sd=np.nan)
    diff = est - gt
    return dict(n=len(gt), mae=np.mean(np.abs(diff)), mae_sd=np.std(np.abs(diff)))


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_rppg_file(df, fname, t0):
    algo_cols = [c for c in df.columns if c.startswith("HR_") and c != "HR_gt"]
    if not algo_cols:
        print(f"  [SKIP] No HR_<algo> column in {fname}")
        return []
    if 'HR_gt' not in df.columns:
        print(f"  [SKIP] No HR_gt in {fname}")
        return []

    results = []
    for algo_col in algo_cols:
        algo = algo_col.replace("HR_", "")
        m = compute_metrics(df['HR_gt'].values, df[algo_col].values)
        m.update(file=fname, algo=algo, metric='HR', unit='BPM')
        results.append(m)
        print(f"  HR {algo}: n={m['n']}  MAE={m['mae']:.2f}±{m['mae_sd']:.2f} BPM")

        if 'SDNN_gt' in df.columns and f'SDNN_{algo}' in df.columns:
            ms = compute_metrics(df['SDNN_gt'].values, df[f'SDNN_{algo}'].values)
            ms.update(file=fname, algo=algo, metric='SDNN', unit='ms')
            results.append(ms)
            print(f"  SDNN {algo}: n={ms['n']}  MAE={ms['mae']:.2f}±{ms['mae_sd']:.2f} ms")

        if 'RMSSD_gt' in df.columns and f'RMSSD_{algo}' in df.columns:
            mr = compute_metrics(df['RMSSD_gt'].values, df[f'RMSSD_{algo}'].values)
            mr.update(file=fname, algo=algo, metric='RMSSD', unit='ms')
            results.append(mr)
            print(f"  RMSSD {algo}: n={mr['n']}  MAE={mr['mae']:.2f}±{mr['mae_sd']:.2f} ms")

    return results


def evaluate_resp_file(df, fname, t0):
    if 'RR_gt' not in df.columns:
        print(f"  [SKIP] No RR_gt in {fname}")
        return []
    if 'RR_BARTULA' not in df.columns:
        print(f"  [SKIP] No RR_BARTULA in {fname}")
        return []

    m = compute_metrics(df['RR_gt'].values, df['RR_BARTULA'].values)
    m.update(file=fname, algo='BARTULA', metric='RR', unit='RPM')
    print(f"  RR BARTULA: n={m['n']}  MAE={m['mae']:.2f}±{m['mae_sd']:.2f} RPM")
    results = [m]

    if 'BBI_gt' in df.columns and 'BBI_BARTULA' in df.columns:
        mb = compute_metrics(df['BBI_gt'].values, df['BBI_BARTULA'].values)
        mb.update(file=fname, algo='BARTULA', metric='BBI', unit='s')
        results.append(mb)
        print(f"  BBI BARTULA: n={mb['n']}  MAE={mb['mae']:.2f}±{mb['mae_sd']:.2f} s")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _is_imu(fname):
    return 'imu' in fname.lower()


def _agg_row(label, unit, rows):
    valid = rows.dropna(subset=['mae'])
    if valid.empty:
        return None
    return dict(label=label, unit=unit, n_files=len(valid),
                mae=valid['mae'].mean(), mae_sd=valid['mae_sd'].mean())


def _print_agg_table(title, rows):
    W = 65
    print(f"\n{title}")
    print("-" * W)
    print(f"  {'ALGO':<14} {'N files':>7}  {'MAE±SD':>14}")
    print("-" * W)
    for row in rows:
        if row is None:
            continue
        unit = row['unit']
        print(f"  {row['label']:<14} {row['n_files']:>7}  "
              f"{row['mae']:.2f}\u00b1{row['mae_sd']:.2f} {unit}")
    print("-" * W)


def _print_hrv_table(title, algo_rows):
    W = 85
    print(f"\n{title}")
    print("-" * W)
    print(f"  {'ALGO':<14} {'N':>4}  {'SDNN MAE±SD':>16}  {'RMSSD MAE±SD':>16}  {'BBI MAE±SD':>14}")
    print("-" * W)
    for algo, metrics in algo_rows:
        sdnn  = metrics.get('SDNN')
        rmssd = metrics.get('RMSSD')
        bbi   = metrics.get('BBI')
        n = (sdnn or rmssd or bbi or {}).get('n_files', 0)

        def _fmt(m, unit, w):
            if m is None or pd.isna(m.get('mae', np.nan)):
                return f"{'—':>{w}}"
            return f"{m['mae']:.2f}\u00b1{m['mae_sd']:.2f} {unit}"

        print(f"  {algo:<14} {n:>4}  "
              f"{_fmt(sdnn,  'ms', 16):>16}  "
              f"{_fmt(rmssd, 'ms', 16):>16}  "
              f"{_fmt(bbi,   's',  14):>14}")
    print("-" * W)


def print_summary(all_results):
    # --- Aggregate tables ---
    df_r = pd.DataFrame(all_results)
    df_r['imu'] = df_r['file'].apply(_is_imu)

    # Table 1 — standard algorithms (no IMU)
    def agg(algo, metric, imu=None):
        sub = df_r[(df_r['algo'] == algo) & (df_r['metric'] == metric)]
        if imu is not None:
            sub = sub[sub['imu'] == imu]
        unit = sub['unit'].iloc[0] if not sub.empty else ''
        label = algo if imu is not False else algo
        return _agg_row(label, unit, sub)

    # Table 1 — Rates: standard algorithms (no motion compensation)
    t1_rows = [r for r in [
        agg('GREEN',    'HR',  imu=False),
        agg('OMIT',     'HR'),
        agg('POS_WANG', 'HR'),
        agg('BARTULA',  'RR'),
    ] if r is not None]

    # Table 2 — Rates: motion / IMU compensation
    t2_rows = [r for r in [
        agg('GREEN', 'HR', imu=True),
        agg('LMS',   'HR'),
    ] if r is not None]

    # Table 3 — HRV + RRV: one row per algorithm
    t3_algos = [
        ('GREEN',    False),
        ('OMIT',     None),
        ('POS_WANG', None),
        ('BARTULA',  None),
    ]
    t3_rows = []
    for algo, imu in t3_algos:
        sdnn  = agg(algo, 'SDNN',  imu=imu)
        rmssd = agg(algo, 'RMSSD', imu=imu)
        bbi   = agg(algo, 'BBI',   imu=imu)
        if any(x is not None for x in [sdnn, rmssd, bbi]):
            t3_rows.append((algo, {'SDNN': sdnn, 'RMSSD': rmssd, 'BBI': bbi}))

    # Table 4 — HRV: motion / IMU compensation
    t4_algos = [('GREEN', True), ('LMS', None)]
    t4_rows = []
    for algo, imu in t4_algos:
        sdnn  = agg(algo, 'SDNN',  imu=imu)
        rmssd = agg(algo, 'RMSSD', imu=imu)
        if any(x is not None for x in [sdnn, rmssd]):
            t4_rows.append((algo, {'SDNN': sdnn, 'RMSSD': rmssd}))

    if t1_rows:
        _print_agg_table("Table 1 — Rate: standard algorithms", t1_rows)
    if t2_rows:
        _print_agg_table("Table 2 — Rate: motion / IMU compensation", t2_rows)
    if t3_rows:
        _print_hrv_table("Table 3 — HRV + RRV: standard algorithms", t3_rows)
    if t4_rows:
        _print_hrv_table("Table 4 — HRV: motion / IMU compensation", t4_rows)


def save_summary_csv(all_results):
    os.makedirs(CSV_OUT, exist_ok=True)
    path = os.path.join(CSV_OUT, "summary_metrics.csv")
    pd.DataFrame(all_results).to_csv(path, index=False)
    print(f"\nSaved summary: {path}")


# ---------------------------------------------------------------------------
# GT interpolation helper
# ---------------------------------------------------------------------------

def _merge_gt(df_sw, df_hw):
    """Interpolate GT metric columns from HW timestamps into df_sw timestamps."""
    if df_hw is None:
        return df_sw
    t_sw = df_sw['timestamp'].values.astype(float)
    t_hw = df_hw['timestamp'].values.astype(float)
    gt_cols = [c for c in df_hw.columns if c not in ('timestamp', 'signal_hw')]
    df_sw = df_sw.copy()
    for col in gt_cols:
        vals = df_hw[col].values.astype(float)
        mask = np.isfinite(vals)
        if mask.sum() < 2:
            df_sw[col] = np.nan
            continue
        interp = np.interp(t_sw, t_hw[mask], vals[mask])
        interp[t_sw < t_hw[mask][0]]  = np.nan
        interp[t_sw > t_hw[mask][-1]] = np.nan
        df_sw[col] = interp
    return df_sw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rPPG/RR recordings from stored GT and algo columns.")
    parser.add_argument('--file', type=str, default=None,
                        help="Single _sw_processed.csv file to evaluate (name only)")
    args = parser.parse_args()

    csv_files = [args.file] if args.file else sorted(
        f for f in os.listdir(RECORDINGS)
        if f.endswith('_sw_processed.csv')
    )

    if not csv_files:
        print("No _sw_processed.csv files found.")
        sys.exit(1)

    all_results = []

    for fname in csv_files:
        fpath = os.path.join(RECORDINGS, fname)
        if not os.path.exists(fpath):
            print(f"[NOT FOUND] {fname}")
            continue

        print(f"\nEvaluating: {fname}")
        try:
            df_sw = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        hw_fname = fname.replace("_sw_processed.csv", "_hw_processed.csv")
        hw_path  = os.path.join(RECORDINGS, hw_fname)
        df_hw    = pd.read_csv(hw_path) if os.path.exists(hw_path) else None

        # merge GT columns (interpolated from HW timestamps) into SW df
        df = _merge_gt(df_sw, df_hw)

        is_resp = 'RR_BARTULA' in df.columns and not any(c.startswith('HR_') for c in df.columns)
        is_rppg = any(c.startswith('HR_') and c != 'HR_gt' for c in df.columns)

        # Skip warmup buffer: 30 s for rPPG, 20 s for respiratory
        warmup = 20.0 if is_resp else 30.0
        t0 = df['timestamp'].iloc[0]
        df = df[df['timestamp'] - t0 >= warmup].reset_index(drop=True)
        if df.empty:
            print(f"  [SKIP] No data after {warmup:.0f} s warmup filter")
            continue

        if is_resp and not is_rppg:
            results = evaluate_resp_file(df, fname, t0)
            plot_resp(df, fname, PLOTS_EVAL, df_hw=df_hw)
        elif is_rppg:
            results = evaluate_rppg_file(df, fname, t0)
            plot_rppg(df, fname, PLOTS_EVAL, df_hw=df_hw)
        else:
            print(f"  [SKIP] Unknown format")
            continue

        all_results.extend(results)

    if not all_results:
        print("\nNo results.")
        return

    print_summary(all_results)


if __name__ == "__main__":
    main()
