"""
evaluate.py — Offline evaluation of rPPG and respiration recordings.

Reads all CSVs from recordings/ and produces:
  - MAE, RMSE, mean bias, Pearson r for each metric
  - Bland-Altman plots
  - Scatter + time-series plots

Usage:
    python evaluate.py                          # evaluate all recordings
    python evaluate.py --file Recording_X.csv  # evaluate one specific file
    python evaluate.py --recompute             # also re-estimate rates via FFT from raw signals
    python evaluate.py --fix                   # recompute metrics from raw signals and overwrite CSVs
    python evaluate.py --fix --file X.csv      # fix one file only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RECORDINGS   = os.path.join(SCRIPT_DIR, "recordings")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")

# ---------------------------------------------------------------------------
# Physiological constants
# ---------------------------------------------------------------------------
HR_LOW_HZ  = 0.75   # 45 BPM
HR_HIGH_HZ = 3.33   # 200 BPM
RR_LOW_HZ  = 0.10   # 6 RPM
RR_HIGH_HZ = 0.70   # 42 RPM

# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def butter_bandpass(signal, fs, low, high, order=4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n  = max(low  / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal)


def estimate_rate_fft(signal, timestamps, f_low, f_high):
    """
    Estimate periodic rate (BPM or RPM) via FFT.

    Parameters
    ----------
    signal     : 1-D array
    timestamps : 1-D array (seconds, same length as signal)
    f_low      : lower frequency bound (Hz)
    f_high     : upper frequency bound (Hz)

    Returns
    -------
    rate : float  (beats or breaths per minute)  or np.nan on failure
    """
    signal = np.asarray(signal, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    # Drop NaNs
    mask = np.isfinite(signal) & np.isfinite(timestamps)
    signal, timestamps = signal[mask], timestamps[mask]
    if len(signal) < 30:
        return np.nan

    # Estimate fs from median sample interval
    dt = np.median(np.diff(timestamps))
    if dt <= 0:
        return np.nan
    fs = 1.0 / dt

    # Need at least 10 s of data for meaningful FFT
    if len(signal) / fs < 10:
        return np.nan

    # Detrend + bandpass
    signal = signal - np.mean(signal)
    try:
        signal = butter_bandpass(signal, fs, f_low, f_high)
    except Exception:
        return np.nan

    # FFT
    n   = len(signal)
    fft = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask_f = (freqs >= f_low) & (freqs <= f_high)
    if not mask_f.any():
        return np.nan

    peak_idx = np.argmax(fft[mask_f])
    dominant_freq = freqs[mask_f][peak_idx]
    return dominant_freq * 60.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(gt, est):
    """
    Compute MAE, RMSE, mean bias (est − gt), and Pearson r.
    Both arrays must be 1-D numeric; NaN pairs are dropped.
    """
    gt  = np.asarray(gt, dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(gt) & np.isfinite(est)
    gt, est = gt[mask], est[mask]

    if len(gt) < 3:
        return dict(n=len(gt), mae=np.nan, rmse=np.nan, bias=np.nan, r=np.nan, p=np.nan)

    diff = est - gt
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    r, p = pearsonr(gt, est) if len(gt) >= 3 else (np.nan, np.nan)
    return dict(n=len(gt), mae=mae, rmse=rmse, bias=bias, r=r, p=p)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def bland_altman(ax, gt, est, label, unit="BPM"):
    """Bland-Altman (mean–difference) plot."""
    gt  = np.asarray(gt, dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(gt) & np.isfinite(est)
    gt, est = gt[mask], est[mask]
    if len(gt) < 3:
        ax.set_title(f"{label} — not enough data")
        return

    mean = (gt + est) / 2.0
    diff = est - gt
    bias = np.mean(diff)
    loa  = 1.96 * np.std(diff)

    ax.scatter(mean, diff, alpha=0.5, s=15, color='steelblue')
    ax.axhline(bias,        color='red',    linestyle='--', lw=1.5, label=f"Bias: {bias:+.2f}")
    ax.axhline(bias + loa,  color='orange', linestyle=':',  lw=1.2, label=f"+1.96σ: {bias+loa:+.2f}")
    ax.axhline(bias - loa,  color='orange', linestyle=':',  lw=1.2, label=f"−1.96σ: {bias-loa:+.2f}")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f"Mean ({unit})")
    ax.set_ylabel(f"Difference est − gt ({unit})")
    ax.set_title(f"Bland-Altman: {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def scatter_plot(ax, gt, est, label, unit="BPM"):
    """Scatter + identity line."""
    gt  = np.asarray(gt, dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(gt) & np.isfinite(est)
    gt, est = gt[mask], est[mask]
    if len(gt) < 3:
        ax.set_title(f"{label} — not enough data")
        return

    ax.scatter(gt, est, alpha=0.5, s=15, color='steelblue')
    lim_min = min(gt.min(), est.min()) - 5
    lim_max = max(gt.max(), est.max()) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1, label='Identity')

    # Regression line
    m, b = np.polyfit(gt, est, 1)
    ax.plot([lim_min, lim_max], [m * lim_min + b, m * lim_max + b],
            'r-', lw=1.5, label=f"y={m:.2f}x+{b:.1f}")

    r, _ = pearsonr(gt, est)
    ax.set_xlabel(f"Ground Truth ({unit})")
    ax.set_ylabel(f"Estimated ({unit})")
    ax.set_title(f"Scatter: {label}  (r={r:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def timeseries_plot(ax, timestamps, gt, est, label_gt, label_est, unit="BPM"):
    """Time-series overlay of gt vs estimate."""
    t0 = timestamps[np.isfinite(timestamps)][0] if np.isfinite(timestamps).any() else 0
    t  = timestamps - t0
    ax.plot(t, gt,  label=label_gt,  color='red',   alpha=0.8, lw=1.2)
    ax.plot(t, est, label=label_est, color='green',  alpha=0.8, lw=1.2, linestyle='--')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(unit)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_rppg_file(df, fname, recompute, out_dir):
    """
    Evaluate one rPPG recording CSV.
    Returns list of metric dicts.
    """
    # Detect algo column
    algo_cols = [c for c in df.columns if c.startswith("HR_") and c != "HR_gt"]
    if not algo_cols:
        print(f"  [SKIP] No HR_<algo> column found in {fname}")
        return []

    algo_col = algo_cols[0]
    algo     = algo_col.replace("HR_", "")

    results = []
    timestamps = df['timestamp'].values

    # --- 1. Stored metrics (computed live with find_peaks) ---
    m_hr = compute_metrics(df['HR_gt'], df[algo_col])
    m_hr['file']   = fname
    m_hr['algo']   = algo
    m_hr['metric'] = 'HR (stored find_peaks)'
    m_hr['unit']   = 'BPM'
    results.append(m_hr)

    for metric in ['SDNN', 'RMSSD']:
        gt_col  = f'{metric}_gt'
        est_col = f'{metric}_{algo}'
        if gt_col in df.columns and est_col in df.columns:
            m = compute_metrics(df[gt_col], df[est_col])
            m['file']   = fname
            m['algo']   = algo
            m['metric'] = f'{metric} (stored find_peaks)'
            m['unit']   = 'ms'
            results.append(m)

    # --- 2. Re-estimated via FFT from raw signals ---
    if recompute and 'signal_hw' in df.columns and 'signal_sw' in df.columns:
        # Segment into ~10-second windows, estimate per window
        window_s = 20
        step_s   = 5
        dt       = np.median(np.diff(timestamps))
        if dt > 0:
            fs_cam = 1.0 / dt
            w = int(window_s * fs_cam)
            s = int(step_s   * fs_cam)
            gt_fft, est_fft = [], []
            for start in range(0, len(df) - w, s):
                seg = df.iloc[start:start + w]
                r_gt  = estimate_rate_fft(seg['signal_hw'], seg['timestamp'], HR_LOW_HZ, HR_HIGH_HZ)
                r_est = estimate_rate_fft(seg['signal_sw'], seg['timestamp'], HR_LOW_HZ, HR_HIGH_HZ)
                gt_fft.append(r_gt)
                est_fft.append(r_est)

            m_fft = compute_metrics(gt_fft, est_fft)
            m_fft['file']   = fname
            m_fft['algo']   = algo
            m_fft['metric'] = 'HR (FFT recomputed)'
            m_fft['unit']   = 'BPM'
            results.append(m_fft)

    # --- 3. Plots ---
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{fname}  —  Algorithm: {algo}", fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_ts  = fig.add_subplot(gs[0, :])   # full-width time series
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_ba  = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis('off')

    timeseries_plot(ax_ts, timestamps, df['HR_gt'].values, df[algo_col].values,
                    'HR ground truth', f'HR {algo}')
    ax_ts.set_title("Heart Rate over Recording")

    scatter_plot(ax_sc, df['HR_gt'], df[algo_col], f'HR {algo}')
    bland_altman(ax_ba, df['HR_gt'], df[algo_col], f'HR {algo}')

    # Metrics text box
    lines = ["Stored metrics (find_peaks):\n"]
    for m in results:
        if 'stored' in m['metric'].lower() or 'fft' in m['metric'].lower():
            lines.append(
                f"{m['metric']}\n"
                f"  n={m['n']}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}\n"
                f"  Bias={m['bias']:+.2f}  r={m['r']:.3f}\n"
            )
    ax_txt.text(0.02, 0.98, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))

    basename = os.path.splitext(fname)[0]
    fig.savefig(os.path.join(out_dir, f"{basename}.png"), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {basename}.png")

    return results


def evaluate_resp_file(df, fname, recompute, out_dir):
    """
    Evaluate one respiration recording CSV.
    Returns list of metric dicts.
    """
    if 'RR_gt' not in df.columns or 'RR_BARTULA' not in df.columns:
        print(f"  [SKIP] Missing RR columns in {fname}")
        return []

    results = []
    timestamps = df['timestamp'].values

    # --- 1. Stored metrics ---
    m_rr = compute_metrics(df['RR_gt'], df['RR_BARTULA'])
    m_rr['file']   = fname
    m_rr['algo']   = 'BARTULA'
    m_rr['metric'] = 'RR (stored find_peaks)'
    m_rr['unit']   = 'RPM'
    results.append(m_rr)

    if 'BRV_gt' in df.columns and 'BRV_BARTULA' in df.columns:
        m_brv = compute_metrics(df['BRV_gt'], df['BRV_BARTULA'])
        m_brv['file']   = fname
        m_brv['algo']   = 'BARTULA'
        m_brv['metric'] = 'BRV (stored find_peaks)'
        m_brv['unit']   = 'ms'
        results.append(m_brv)

    # --- 2. Re-estimated via FFT ---
    if recompute and 'signal_hw' in df.columns and 'signal_sw' in df.columns:
        window_s = 30
        step_s   = 10
        dt       = np.median(np.diff(timestamps))
        if dt > 0:
            fs_cam = 1.0 / dt
            w = int(window_s * fs_cam)
            s = int(step_s   * fs_cam)
            gt_fft, est_fft = [], []
            for start in range(0, len(df) - w, s):
                seg = df.iloc[start:start + w]
                r_gt  = estimate_rate_fft(seg['signal_hw'], seg['timestamp'], RR_LOW_HZ, RR_HIGH_HZ)
                r_est = estimate_rate_fft(seg['signal_sw'], seg['timestamp'], RR_LOW_HZ, RR_HIGH_HZ)
                gt_fft.append(r_gt)
                est_fft.append(r_est)

            m_fft = compute_metrics(gt_fft, est_fft)
            m_fft['file']   = fname
            m_fft['algo']   = 'BARTULA'
            m_fft['metric'] = 'RR (FFT recomputed)'
            m_fft['unit']   = 'RPM'
            results.append(m_fft)

    # --- 3. Plots ---
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{fname}  —  Respiration (Bartula)", fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_ts  = fig.add_subplot(gs[0, :])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_ba  = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis('off')

    timeseries_plot(ax_ts, timestamps, df['RR_gt'].values, df['RR_BARTULA'].values,
                    'RR ground truth', 'RR Bartula', unit='RPM')
    ax_ts.set_title("Respiration Rate over Recording")

    scatter_plot(ax_sc, df['RR_gt'], df['RR_BARTULA'], 'RR Bartula', unit='RPM')
    bland_altman(ax_ba, df['RR_gt'], df['RR_BARTULA'], 'RR Bartula', unit='RPM')

    lines = ["Stored metrics (find_peaks):\n"]
    for m in results:
        lines.append(
            f"{m['metric']}\n"
            f"  n={m['n']}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}\n"
            f"  Bias={m['bias']:+.2f}  r={m['r']:.3f}\n"
        )
    ax_txt.text(0.02, 0.98, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))

    basename = os.path.splitext(fname)[0]
    fig.savefig(os.path.join(out_dir, f"{basename}.png"), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {basename}.png")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results):
    """Print a formatted summary table to stdout."""
    cols = ['file', 'algo', 'metric', 'unit', 'n', 'mae', 'rmse', 'bias', 'r']
    df = pd.DataFrame(all_results, columns=cols)

    print("\n" + "=" * 95)
    print(f"{'FILE':<42} {'ALGO':<10} {'METRIC':<30} {'N':>4} {'MAE':>6} {'RMSE':>6} {'BIAS':>7} {'r':>6}")
    print("=" * 95)

    for _, row in df.iterrows():
        fname_short = row['file'][:40] if len(row['file']) > 40 else row['file']
        mae  = f"{row['mae']:.2f}"  if pd.notna(row['mae'])  else "  N/A"
        rmse = f"{row['rmse']:.2f}" if pd.notna(row['rmse']) else "  N/A"
        bias = f"{row['bias']:+.2f}" if pd.notna(row['bias']) else "   N/A"
        r    = f"{row['r']:.3f}"   if pd.notna(row['r'])    else "  N/A"
        unit = row['unit']
        print(f"{fname_short:<42} {row['algo']:<10} {row['metric']:<30} {int(row['n']):>4}"
              f" {mae:>6}{unit[:3]:3} {rmse:>6} {bias:>7} {r:>6}")

    print("=" * 95)


def save_summary_csv(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary_metrics.csv")
    pd.DataFrame(all_results).to_csv(path, index=False)
    print(f"\nSaved summary CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate rPPG / respiration recordings.")
    parser.add_argument('--file',      type=str,  default=None,
                        help="Evaluate a single file (name only, must be in recordings/)")
    parser.add_argument('--recompute', action='store_true',
                        help="Re-estimate rates via FFT from raw signals")
    args = parser.parse_args()

    out_dir = RESULTS_DIR

    # Collect CSV files
    if args.file:
        csv_files = [args.file]
    else:
        csv_files = sorted([f for f in os.listdir(RECORDINGS) if f.endswith('.csv')])

    if not csv_files:
        print("No CSV files found in recordings/")
        sys.exit(1)

    all_results = []

    for fname in csv_files:
        fpath = os.path.join(RECORDINGS, fname)
        if not os.path.exists(fpath):
            print(f"[NOT FOUND] {fname}")
            continue

        print(f"\nEvaluating: {fname}")
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [ERROR] Could not read: {e}")
            continue

        # Detect recording type
        is_resp = 'RR_gt' in df.columns
        is_rppg = 'HR_gt' in df.columns

        if not is_rppg and not is_resp:
            print(f"  [SKIP] Unknown format (no HR_gt or RR_gt column)")
            continue

        if is_rppg:
            results = evaluate_rppg_file(df, fname, args.recompute, out_dir)
        else:
            results = evaluate_resp_file(df, fname, args.recompute, out_dir)

        all_results.extend(results)

    if not all_results:
        print("\nNo results to display.")
        return

    print_summary(all_results)
    save_summary_csv(all_results, out_dir)
    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    main()
