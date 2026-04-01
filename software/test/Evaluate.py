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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RECORDINGS  = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_OUT     = os.path.join(RESULTS_DIR, "csv_raw")


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

    results = []
    for algo_col in algo_cols:
        algo = algo_col.replace("HR_", "")
        gt   = df['HR_gt'].values
        est  = df[algo_col].values
        m    = compute_metrics(gt, est)
        m.update(file=fname, algo=algo, metric='HR', unit='BPM')
        results.append(m)
        print(f"  HR {algo}: n={m['n']}  MAE={m['mae']:.2f}±{m['mae_sd']:.2f}  "
              f"RMSE={m['rmse']:.2f}  Bias={m['bias']:+.2f}  PCC={m['r']:.3f}")
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
    )


def _print_agg_table(title, rows):
    W = 80
    print(f"\n{title}")
    print("-" * W)
    print(f"  {'ALGO':<14} {'N files':>7}  {'MAE±SD':>14}  {'RMSE':>6}  {'BIAS':>7}  {'PCC':>6}")
    print("-" * W)
    for row in rows:
        if row is None:
            continue
        unit = row['unit']
        print(f"  {row['label']:<14} {row['n_files']:>7}  "
              f"{row['mae']:.2f}±{row['mae_sd']:.2f}{unit[:3]:>3}  "
              f"{row['rmse']:>6.2f}  {row['bias']:>+7.2f}  {row['r']:>6.3f}")
    print("-" * W)


def print_summary(all_results):
    print("\n" + "=" * 105)
    print(f"{'FILE':<45} {'ALGO':<12} {'M':<3} {'N':>5}  "
          f"{'MAE±SD':>16}  {'RMSE':>6}  {'BIAS':>7}  {'PCC':>6}")
    print("=" * 105)

    for m in all_results:
        fname_short = m['file'][:43] if len(m['file']) > 43 else m['file']
        unit   = m['unit']
        mae_s  = f"{m['mae']:.2f}±{m['mae_sd']:.2f}" if pd.notna(m['mae'])  else "N/A"
        rmse_s = f"{m['rmse']:.2f}"                   if pd.notna(m['rmse']) else "N/A"
        bias_s = f"{m['bias']:+.2f}"                  if pd.notna(m['bias']) else "N/A"
        r_s    = f"{m['r']:.3f}"                      if pd.notna(m['r'])    else "N/A"
        print(f"{fname_short:<45} {m['algo']:<12} {m['metric']:<3} {int(m['n']):>5}  "
              f"{mae_s:>13}{unit[:3]}  {rmse_s:>6}  {bias_s:>7}  {r_s:>6}")

    print("=" * 105)

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
