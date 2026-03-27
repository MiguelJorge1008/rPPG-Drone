"""
evaluate.py — rPPG HR estimation evaluation.

Expected CSV columns:
  timestamp   : float or datetime
  HR_gt       : float  Ground truth HR in BPM (oximeter)
  HR_GREEN    : float  (optional)
  HR_OMIT     : float  (optional)
  HR_POS_WANG : float  (optional)
  HR_LMS      : float  (optional)
  roi_mode    : str    FOREHEAD | FACE | MULTI  (optional)
  source      : str    drone | webcam           (optional)

Usage:
  python evaluate.py results.csv
  python evaluate.py results.csv --plot
  python evaluate.py results.csv --save
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── Algorithms to evaluate (auto-detected from CSV columns) ───────────────────
ALL_ALGOS = ["HR_GREEN", "HR_OMIT", "HR_POS_WANG", "HR_LMS"]

ALGO_LABELS = {
    "HR_GREEN":    "GREEN",
    "HR_OMIT":     "OMIT",
    "HR_POS_WANG": "POS_WANG",
    "HR_LMS":      "LMS",
}


# ── Metric functions ──────────────────────────────────────────────────────────

def mae(gt, est):
    valid = ~(np.isnan(gt) | np.isnan(est))
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(gt[valid] - est[valid])))


def rmse(gt, est):
    valid = ~(np.isnan(gt) | np.isnan(est))
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((gt[valid] - est[valid]) ** 2)))


def pcc(gt, est):
    valid = ~(np.isnan(gt) | np.isnan(est))
    if valid.sum() < 2:
        return np.nan
    r, _ = pearsonr(gt[valid], est[valid])
    return float(r)


def compute_metrics(gt, est):
    return {"MAE": mae(gt, est), "RMSE": rmse(gt, est), "PCC": pcc(gt, est)}


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(title, results, metric, fmt=".2f", groups=None):
    """
    results : dict  { algo_col: { group: {"MAE":..,"RMSE":..,"PCC":..} } }
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    algos    = list(results.keys())
    col_w    = 10
    grp_cols = groups if groups else ["Overall"]

    header = f"{'Algorithm':<12}" + "".join(f"{g:>{col_w}}" for g in grp_cols)
    print(header)
    print("-" * len(header))

    for col in algos:
        row = f"{ALGO_LABELS.get(col, col):<12}"
        for g in grp_cols:
            val = results[col].get(g, {}).get(metric, np.nan)
            row += f"{val:>{col_w}{fmt}}" if not np.isnan(val) else f"{'—':>{col_w}}"
        print(row)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(df, group_col=None):
    """
    Compute MAE / RMSE / PCC for each algorithm.
    If group_col is given, also break down by group values.

    Returns dict: { algo_col: { group_label: metrics_dict } }
    """
    algo_cols = [c for c in ALL_ALGOS if c in df.columns]
    gt        = df["HR_gt"].values.astype(float)

    groups = sorted(df[group_col].dropna().unique()) if group_col else []

    results = {}
    for col in algo_cols:
        est             = df[col].values.astype(float)
        results[col]    = {"Overall": compute_metrics(gt, est)}
        for g in groups:
            mask = df[group_col] == g
            results[col][str(g)] = compute_metrics(gt[mask], est[mask])

    return results, algo_cols, [str(g) for g in groups]


# ── Optional plot ─────────────────────────────────────────────────────────────

def plot_results(df):
    import matplotlib.pyplot as plt

    algo_cols = [c for c in ALL_ALGOS if c in df.columns]
    gt        = df["HR_gt"].values.astype(float)
    n         = len(algo_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, col in zip(axes[0], algo_cols):
        est   = df[col].values.astype(float)
        valid = ~(np.isnan(gt) | np.isnan(est))
        ax.scatter(gt[valid], est[valid], alpha=0.6, s=20)
        lims = [min(gt[valid].min(), est[valid].min()) - 5,
                max(gt[valid].max(), est[valid].max()) + 5]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Ground Truth (BPM)")
        ax.set_ylabel("Estimated (BPM)")
        ax.set_title(ALGO_LABELS.get(col, col))
        mae_v = mae(gt[valid], est[valid])
        pcc_v = pcc(gt[valid], est[valid])
        ax.text(0.05, 0.92, f"MAE={mae_v:.2f}  PCC={pcc_v:.2f}",
                transform=ax.transAxes, fontsize=8)

    fig.suptitle("Estimated vs Ground Truth HR")
    fig.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="rPPG evaluation script")
    parser.add_argument("csv",          help="Path to results CSV")
    parser.add_argument("--plot",       action="store_true", help="Show scatter plots")
    parser.add_argument("--save",       action="store_true", help="Save metrics to CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {"HR_gt"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        sys.exit(1)

    algo_cols = [c for c in ALL_ALGOS if c in df.columns]
    if not algo_cols:
        print(f"ERROR: no algorithm columns found. Expected one of: {ALL_ALGOS}")
        sys.exit(1)

    print(f"\nLoaded {len(df)} samples from '{args.csv}'")
    print(f"Algorithms found : {[ALGO_LABELS[c] for c in algo_cols]}")
    if "roi_mode" in df.columns:
        print(f"ROI modes found  : {sorted(df['roi_mode'].dropna().unique())}")
    if "source" in df.columns:
        print(f"Sources found    : {sorted(df['source'].dropna().unique())}")

    # ── Overall results ───────────────────────────────────────────────────────
    results_overall, _, _ = evaluate(df)
    for metric in ("MAE", "RMSE", "PCC"):
        fmt = ".3f" if metric == "PCC" else ".2f"
        unit = "" if metric == "PCC" else " (BPM)"
        print_table(f"{metric}{unit} — Overall", results_overall, metric, fmt)

    # ── Breakdown by roi_mode ─────────────────────────────────────────────────
    if "roi_mode" in df.columns:
        results_roi, _, grps = evaluate(df, group_col="roi_mode")
        for metric in ("MAE", "RMSE", "PCC"):
            fmt = ".3f" if metric == "PCC" else ".2f"
            unit = "" if metric == "PCC" else " (BPM)"
            print_table(f"{metric}{unit} — by ROI mode",
                        results_roi, metric, fmt,
                        groups=grps + ["Overall"])

    # ── Breakdown by source ───────────────────────────────────────────────────
    if "source" in df.columns:
        results_src, _, grps = evaluate(df, group_col="source")
        for metric in ("MAE", "RMSE", "PCC"):
            fmt = ".3f" if metric == "PCC" else ".2f"
            unit = "" if metric == "PCC" else " (BPM)"
            print_table(f"{metric}{unit} — by source",
                        results_src, metric, fmt,
                        groups=grps + ["Overall"])

    # ── Save metrics ──────────────────────────────────────────────────────────
    if args.save:
        rows = []
        for col in algo_cols:
            r = results_overall[col]["Overall"]
            rows.append({
                "algorithm": ALGO_LABELS[col],
                "MAE":  round(r["MAE"],  2),
                "RMSE": round(r["RMSE"], 2),
                "PCC":  round(r["PCC"],  3),
            })
        out = args.csv.replace(".csv", "_metrics.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nMetrics saved to '{out}'")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        plot_results(df)


if __name__ == "__main__":
    main()
