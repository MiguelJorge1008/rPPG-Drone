"""
evaluate.py — Evaluate rPPG / respiration recordings.

Reads HR_gt + HR_<algo>  (or RR_gt + RR_BARTULA) directly from each CSV
and computes:  MAE ± SD,  RMSE,  Bias,  PCC

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
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RECORDINGS  = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_OUT     = os.path.join(RESULTS_DIR, "csv_raw")
PLOTS_EVAL  = os.path.join(RESULTS_DIR, "plots_eval")
ACCURACY_BPM_THRESHOLD = 5    # ±5 bpm — Bartula 2013 / standard rPPG threshold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt, est):
    gt  = np.asarray(gt,  dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(gt) & np.isfinite(est)
    gt, est = gt[mask], est[mask]

    if len(gt) < 3:
        return dict(n=len(gt), mae=np.nan, mae_sd=np.nan,
                    rmse=np.nan, bias=np.nan, r=np.nan, p=np.nan)

    diff   = est - gt
    mae    = np.mean(np.abs(diff))
    mae_sd = np.std(np.abs(diff))
    rmse   = np.sqrt(np.mean(diff ** 2))
    bias   = np.mean(diff)
    r, p   = pearsonr(gt, est)
    return dict(n=len(gt), mae=mae, mae_sd=mae_sd,
                rmse=rmse, bias=bias, r=r, p=p)


# NOTE: This is the Wang 2017 narrow-window SNR. It is NOT the same as the
# broadband SNR in ROIExtraction.py (which is used for region ranking).
def compute_snr(df):
    """Compute spectral SNR for a rPPG signal (Wang 2017).

    Estimates signal-to-noise ratio using a narrow frequency window centred on
    the ground-truth heart rate, with a Hann-windowed power spectrum.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``signal_sw`` (rPPG signal), ``timestamp``
        (seconds, float), and ``HR_gt`` (ground-truth heart rate in BPM).

    Returns
    -------
    float
        SNR in dB, or ``np.nan`` if guards are triggered (too short, flat
        signal, missing GT, or degenerate spectrum).
    """
    try:
        # 1. Extract signal
        signal = df['signal_sw'].dropna().values
        if len(signal) < 2 or np.std(signal) < 1e-9:
            return np.nan

        # 2. Estimate FPS from timestamps
        ts_all = df['timestamp'].values.astype(float)
        ts_valid = ts_all[~np.isnan(ts_all)]
        if len(ts_valid) < 2:
            return np.nan
        diffs = np.diff(ts_valid)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return np.nan
        fps = 1.0 / np.median(diffs)

        # 3. Duration guard
        ts = ts_valid
        if ts[-1] - ts[0] < 30.0:
            return np.nan

        # 4. Ground-truth frequency
        hr_gt_valid = df['HR_gt'].values[np.isfinite(df['HR_gt'].values)]
        if len(hr_gt_valid) == 0:
            return np.nan
        f0 = np.nanmean(hr_gt_valid) / 60.0
        if np.isnan(f0):
            return np.nan

        # 5. Zero-mean, Hann window, FFT
        signal = signal - np.mean(signal)
        signal_windowed = signal * np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal_windowed)) ** 2
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

        # 6. Band power [0.75, 4.0 Hz]
        band_mask = (freqs >= 0.75) & (freqs <= 4.0)
        total_power = np.sum(spectrum[band_mask])
        if total_power == 0:
            return np.nan

        # 7. Signal window and noise
        f_lo = max(0.75, f0 - 0.1)
        f_hi = min(4.0,  f0 + 0.1)
        sig_mask = (freqs >= f_lo) & (freqs <= f_hi)
        P_signal = np.sum(spectrum[sig_mask])
        P_noise  = total_power - P_signal
        if P_noise <= 0:
            return np.nan

        # 8. SNR in dB
        return float(10.0 * np.log10(P_signal / P_noise))

    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_rppg_file(df, fname):
    algo_cols = [c for c in df.columns if c.startswith("HR_") and c != "HR_gt"]
    if not algo_cols:
        print(f"  [SKIP] No HR_<algo> column in {fname}")
        return []
    if 'HR_gt' not in df.columns:
        print(f"  [SKIP] No HR_gt in {fname} — run reprocess_gt.py first")
        return []

    snr_val = compute_snr(df)   # computed once — signal_sw is file-level, not per-algo

    results = []
    for algo_col in algo_cols:
        algo = algo_col.replace("HR_", "")
        gt   = df['HR_gt'].values
        est  = df[algo_col].values
        m    = compute_metrics(gt, est)
        valid_mask = np.isfinite(gt) & np.isfinite(est)
        if valid_mask.sum() >= 3:
            acc_val = float(np.mean(np.abs(est[valid_mask] - gt[valid_mask])
                                    <= ACCURACY_BPM_THRESHOLD) * 100.0)
        else:
            acc_val = np.nan
        m['snr_mean']     = snr_val
        m['accuracy_pct'] = acc_val
        m.update(file=fname, algo=algo, metric='HR', unit='BPM')
        plot_eval_rppg(df, fname, algo, algo_col, m)
        results.append(m)
        snr_s = f"{snr_val:.2f}dB" if pd.notna(snr_val) else "N/A"
        acc_s = f"{acc_val:.1f}%"  if pd.notna(acc_val)  else "N/A"
        print(f"  HR {algo}: n={m['n']}  MAE={m['mae']:.2f}±{m['mae_sd']:.2f}  "
              f"RMSE={m['rmse']:.2f}  Bias={m['bias']:+.2f}  PCC={m['r']:.3f}  "
              f"SNR={snr_s}  Acc={acc_s}")
    return results


def evaluate_resp_file(df, fname):
    if 'RR_gt' not in df.columns:
        print(f"  [SKIP] No RR_gt in {fname} — run reprocess_gt.py first")
        return []
    if 'RR_BARTULA' not in df.columns:
        print(f"  [SKIP] No RR_BARTULA in {fname}")
        return []

    gt  = df['RR_gt'].values
    est = df['RR_BARTULA'].values
    m   = compute_metrics(gt, est)
    m.update(file=fname, algo='BARTULA', metric='RR', unit='RPM')
    print(f"  RR BARTULA: n={m['n']}  MAE={m['mae']:.2f}±{m['mae_sd']:.2f}  "
          f"RMSE={m['rmse']:.2f}  Bias={m['bias']:+.2f}  PCC={m['r']:.3f}")
    return [m]


def plot_eval_rppg(df, fname, algo, algo_col, metrics):
    """Generate a 2-row evaluation PNG for one recording × algorithm pair.

    Parameters
    ----------
    df : pd.DataFrame
        Full recording DataFrame.
    fname : str
        CSV filename (used for title and output PNG stem).
    algo : str
        Algorithm name, e.g. "GREEN".
    algo_col : str
        Column name in df, e.g. "HR_GREEN".
    metrics : dict
        Result dict with keys: mae, mae_sd, rmse, bias, r, snr_mean, accuracy_pct.
    """
    t   = df['timestamp'].values.astype(float)
    t   = t - t[0]
    gt  = df['HR_gt'].values.astype(float)
    est = df[algo_col].values.astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{fname}  [{algo}]", fontsize=11, fontweight='bold')

    # --- Row 1: BPM over time ---
    ax1.plot(t, gt,  color='red',   lw=1.0, linestyle='--', label='HR_gt')
    ax1.plot(t, est, color='green', lw=1.0, linestyle='-',  label=f'HR_{algo}')
    ax1.set_ylabel('HR (bpm)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Heart Rate over Time')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Row 2: Bland-Altman ---
    mask = np.isfinite(gt) & np.isfinite(est)
    if mask.sum() < 3 or not pd.notna(metrics.get('bias')):
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color='gray')
        ax2.set_title('Bland-Altman — Insufficient data')
    else:
        means = (gt[mask] + est[mask]) / 2.0
        diffs = est[mask] - gt[mask]
        bias  = np.mean(diffs)
        sd    = np.std(diffs, ddof=1)
        loa_u = bias + 1.96 * sd
        loa_l = bias - 1.96 * sd
        ax2.scatter(means, diffs, s=12, alpha=0.6, color='steelblue')
        ax2.axhline(bias,  color='red',    lw=1.5, linestyle='-',
                    label=f'Bias={bias:+.2f}')
        ax2.axhline(loa_u, color='orange', lw=1.0, linestyle='--',
                    label=f'+1.96SD={loa_u:+.2f}')
        ax2.axhline(loa_l, color='orange', lw=1.0, linestyle='--',
                    label=f'\u22121.96SD={loa_l:+.2f}')
        ax2.axhline(0.0,   color='black',  lw=0.5, linestyle=':')
        ax2.set_xlabel('Mean of GT and Estimated (bpm)')
        ax2.set_ylabel('Estimated \u2212 GT (bpm)')
        ax2.set_title(f'Bland-Altman \u2014 {algo}')
        ax2.legend(fontsize=7, loc='upper right')
        ax2.grid(True, alpha=0.3)

    # --- Metrics text box ---
    snr_s = f"{metrics['snr_mean']:.2f} dB" if pd.notna(metrics.get('snr_mean')) else 'N/A'
    acc_s = f"{metrics['accuracy_pct']:.1f}%" if pd.notna(metrics.get('accuracy_pct')) else 'N/A'
    mae_s = f"{metrics['mae']:.2f}" if pd.notna(metrics.get('mae')) else 'N/A'
    txt = (f"MAE={mae_s}\u00b1{metrics['mae_sd']:.2f}  RMSE={metrics['rmse']:.2f}  "
           f"Bias={metrics['bias']:+.2f}\nPCC={metrics['r']:.3f}  "
           f"SNR={snr_s}  Acc={acc_s}")
    ax1.text(0.99, 0.02, txt, ha='right', va='bottom',
             transform=ax1.transAxes, fontsize=7.5,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round'))

    # --- Save ---
    os.makedirs(PLOTS_EVAL, exist_ok=True)
    stem    = fname.replace('.csv', '')
    outpath = os.path.join(PLOTS_EVAL, f"{stem}_{algo}.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {os.path.basename(outpath)}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _is_imu(fname):
    return 'imu' in fname.lower()


def _agg_row(label, unit, rows):
    valid = rows.dropna(subset=['mae'])
    if valid.empty:
        return None
    return dict(
        label=label, unit=unit, n_files=len(valid),
        mae=valid['mae'].mean(), mae_sd=valid['mae_sd'].mean(),
        rmse=valid['rmse'].mean(), bias=valid['bias'].mean(),
        r=valid['r'].mean(),
        snr_mean=rows.dropna(subset=['snr_mean'])['snr_mean'].mean() if 'snr_mean' in rows.columns and not rows.dropna(subset=['snr_mean']).empty else np.nan,
        accuracy_pct=rows.dropna(subset=['accuracy_pct'])['accuracy_pct'].mean() if 'accuracy_pct' in rows.columns and not rows.dropna(subset=['accuracy_pct']).empty else np.nan,
    )


def _print_agg_table(title, rows):
    W = 105
    print(f"\n{title}")
    print("-" * W)
    print(f"  {'ALGO':<14} {'N files':>7}  {'MAE±SD':>14}  {'RMSE':>6}  {'BIAS':>7}  {'PCC':>6}  {'SNR':>8}  {'Acc%':>6}")
    print("-" * W)
    for row in rows:
        if row is None:
            continue
        unit = row['unit']
        snr_s = f"{row['snr_mean']:.2f}dB" if pd.notna(row.get('snr_mean')) else "N/A"
        acc_s = f"{row['accuracy_pct']:.1f}%" if pd.notna(row.get('accuracy_pct')) else "N/A"
        print(f"  {row['label']:<14} {row['n_files']:>7}  "
              f"{row['mae']:.2f}\u00b1{row['mae_sd']:.2f}{unit[:3]:>3}  "
              f"{row['rmse']:>6.2f}  {row['bias']:>+7.2f}  {row['r']:>6.3f}  "
              f"{snr_s:>8}  {acc_s:>6}")
    print("-" * W)


def print_summary(all_results):
    print("\n" + "=" * 125)
    print(f"{'FILE':<45} {'ALGO':<12} {'M':<3} {'N':>5}  "
          f"{'MAE\u00b1SD':>16}  {'RMSE':>6}  {'BIAS':>7}  {'PCC':>6}  {'SNR':>8}  {'Acc%':>6}")
    print("=" * 125)

    for m in all_results:
        fname_short = m['file'][:43] if len(m['file']) > 43 else m['file']
        unit   = m['unit']
        mae_s  = f"{m['mae']:.2f}±{m['mae_sd']:.2f}" if pd.notna(m['mae'])  else "N/A"
        rmse_s = f"{m['rmse']:.2f}"                   if pd.notna(m['rmse']) else "N/A"
        bias_s = f"{m['bias']:+.2f}"                  if pd.notna(m['bias']) else "N/A"
        r_s    = f"{m['r']:.3f}"                      if pd.notna(m['r'])    else "N/A"
        snr_s  = f"{m['snr_mean']:.2f}dB"             if pd.notna(m.get('snr_mean'))     else "N/A"
        acc_s  = f"{m['accuracy_pct']:.1f}%"          if pd.notna(m.get('accuracy_pct')) else "N/A"
        print(f"{fname_short:<45} {m['algo']:<12} {m['metric']:<3} {int(m['n']):>5}  "
              f"{mae_s:>13}{unit[:3]}  {rmse_s:>6}  {bias_s:>7}  {r_s:>6}  "
              f"{snr_s:>8}  {acc_s:>6}")

    print("=" * 125)

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

    t1_rows = [
        agg('GREEN',   'HR', imu=False),
        agg('OMIT',    'HR'),
        agg('POS_WANG','HR'),
        agg('BARTULA', 'RR'),
    ]
    t1_rows = [r for r in t1_rows if r is not None]

    t2_rows = [
        agg('GREEN',   'HR', imu=True),
        agg('LMS',     'HR'),
    ]
    t2_rows = [r for r in t2_rows if r is not None]

    if t1_rows:
        _print_agg_table("Table 1 — Standard algorithms", t1_rows)
    if t2_rows:
        _print_agg_table("Table 2 — IMU / Motion compensation", t2_rows)


def save_summary_csv(all_results):
    os.makedirs(CSV_OUT, exist_ok=True)
    path = os.path.join(CSV_OUT, "summary_metrics.csv")
    pd.DataFrame(all_results).to_csv(path, index=False)
    print(f"\nSaved summary: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rPPG/RR recordings from stored GT and algo columns.")
    parser.add_argument('--file', type=str, default=None,
                        help="Single file to evaluate (name only)")
    args = parser.parse_args()

    csv_files = [args.file] if args.file else sorted(
        f for f in os.listdir(RECORDINGS) if f.endswith('.csv')
    )

    if not csv_files:
        print("No CSV files found.")
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
            print(f"  [ERROR] {e}")
            continue

        is_resp = 'RR_BARTULA' in df.columns
        is_rppg = any(c.startswith('HR_') for c in df.columns)

        if is_resp and not is_rppg:
            results = evaluate_resp_file(df, fname)
        elif is_rppg:
            results = evaluate_rppg_file(df, fname)
        else:
            print(f"  [SKIP] Unknown format")
            continue

        all_results.extend(results)

    if not all_results:
        print("\nNo results.")
        return

    print_summary(all_results)
    save_summary_csv(all_results)


if __name__ == "__main__":
    main()
