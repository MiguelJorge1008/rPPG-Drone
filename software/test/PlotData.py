"""
plot_signals.py — Plot raw signals and computed metrics over time for each recording.

Generates one PNG per CSV with 3 rows:
  - Row 1: signal_hw vs signal_sw (both normalised to z-score for comparison)
  - Row 2: rate over time (HR_gt vs HR_algo  or  RR_gt vs RR_BARTULA)
  - Row 3: variability over time (SDNN/RMSSD  or  BRV)

Usage:
    python plot_signals.py                          # all recordings (raw)
    python plot_signals.py --source clean           # recordings_cleaned/
    python plot_signals.py --file Recording_X.csv   # single file
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalise(s):
    """Z-score normalisation, ignoring NaNs."""
    s = np.asarray(s, dtype=float)
    std = np.nanstd(s)
    return (s - np.nanmean(s)) / std if std > 0 else s - np.nanmean(s)


SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_RAW     = os.path.join(SCRIPT_DIR, "data")
RECORDINGS_CLEANED = os.path.join(SCRIPT_DIR, "data_cleaned")
RESULTS_BASE       = os.path.join(SCRIPT_DIR, "results")
PLOTS_RAW          = os.path.join(RESULTS_BASE, "plots_raw")
PLOTS_CLEAN        = os.path.join(RESULTS_BASE, "plots_clean")


# ---------------------------------------------------------------------------
# Plot one file
# ---------------------------------------------------------------------------

def plot_rppg(df, fname, out_dir):
    algo_cols = [c for c in df.columns if c.startswith("HR_") and c != "HR_gt"]
    if not algo_cols:
        print(f"  [SKIP] No HR_<algo> column in {fname}")
        return
    algo_col = algo_cols[0]
    algo     = algo_col.replace("HR_", "")

    t = df["timestamp"].values.astype(float)
    t = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(fname, fontsize=11, fontweight="bold")

    # --- Row 0: raw signals (normalised for comparison) ---
    ax = axes[0]
    ax.plot(t, normalise(df["signal_hw"]), color="red",   lw=0.7, label="signal_hw  (hardware PPG)")
    ax.plot(t, normalise(df["signal_sw"]), color="green", lw=0.7, label="signal_sw  (camera BVP)", alpha=0.8)
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Raw signals (z-score normalised)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 1: HR over time ---
    ax = axes[1]
    ax.plot(t, df["HR_gt"],  color="red",   lw=1.2, label="HR_gt  (hardware)")
    ax.plot(t, df[algo_col], color="green", lw=1.2, label=f"{algo_col}  (camera)", alpha=0.8)
    ax.set_ylabel("BPM")
    ax.set_title("Heart Rate over time")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 2: HRV over time ---
    ax = axes[2]
    has_hrv = f"SDNN_gt" in df.columns and f"SDNN_{algo}" in df.columns
    if has_hrv:
        ax.plot(t, df["SDNN_gt"],          color="red",       lw=1.2, label="SDNN_gt")
        ax.plot(t, df[f"SDNN_{algo}"],     color="green",     lw=1.2, label=f"SDNN_{algo}", alpha=0.8)
        ax.plot(t, df["RMSSD_gt"],         color="darkred",   lw=1.0, label="RMSSD_gt",   linestyle="--")
        ax.plot(t, df[f"RMSSD_{algo}"],    color="darkgreen", lw=1.0, label=f"RMSSD_{algo}", alpha=0.8, linestyle="--")
        ax.set_ylabel("ms")
        ax.set_title("HRV — SDNN / RMSSD over time")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.set_title("HRV — no data")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    _save(fig, fname, out_dir)


def plot_resp(df, fname, out_dir):
    t = df["timestamp"].values.astype(float)
    t = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(fname, fontsize=11, fontweight="bold")

    # --- Row 0: raw signals (normalised for comparison) ---
    ax = axes[0]
    ax.plot(t, normalise(df["signal_hw"]), color="blue", lw=0.7, label="signal_hw  (hardware resp)")
    ax.plot(t, normalise(df["signal_sw"]), color="cyan", lw=0.7, label="signal_sw  (camera resp)", alpha=0.8)
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Raw signals (z-score normalised)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 1: RR over time ---
    ax = axes[1]
    ax.plot(t, df["RR_gt"],      color="blue", lw=1.2, label="RR_gt  (hardware)")
    ax.plot(t, df["RR_BARTULA"], color="cyan", lw=1.2, label="RR_BARTULA  (camera)", alpha=0.8)
    ax.set_ylabel("RPM")
    ax.set_title("Respiration Rate over time")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 2: BRV over time ---
    ax = axes[2]
    if "BRV_gt" in df.columns and "BRV_BARTULA" in df.columns:
        ax.plot(t, df["BRV_gt"],      color="blue", lw=1.2, label="BRV_gt")
        ax.plot(t, df["BRV_BARTULA"], color="cyan", lw=1.2, label="BRV_BARTULA", alpha=0.8)
        ax.set_ylabel("ms")
        ax.set_title("BRV over time")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.set_title("BRV — no data")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)")

    _save(fig, fname, out_dir)


def _save(fig, fname, out_dir):
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, fname.replace(".csv", ".png"))
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot signals and metrics over time.")
    parser.add_argument("--file",   type=str, default=None,
                        help="Single file to plot (name only)")
    parser.add_argument("--source", type=str, default="raw",
                        choices=["raw", "clean"],
                        help="'raw' = recordings/   'clean' = recordings_cleaned/")
    args = parser.parse_args()

    if args.source == "clean":
        if not os.path.isdir(RECORDINGS_CLEANED):
            print("ERROR: recordings_cleaned/ not found. Run datacleaning.py first.")
            sys.exit(1)
        rec_dir = RECORDINGS_CLEANED
        out_dir = PLOTS_CLEAN
    else:
        rec_dir = RECORDINGS_RAW
        out_dir = PLOTS_RAW

    print(f"Source : {rec_dir}")
    print(f"Output : {out_dir}\n")

    csv_files = [args.file] if args.file else sorted(
        f for f in os.listdir(rec_dir) if f.endswith(".csv")
    )

    for fname in csv_files:
        fpath = os.path.join(rec_dir, fname)
        if not os.path.exists(fpath):
            print(f"[NOT FOUND] {fname}")
            continue

        print(f"Plotting: {fname}")
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        if "HR_gt" in df.columns:
            plot_rppg(df, fname, out_dir)
        elif "RR_gt" in df.columns:
            plot_resp(df, fname, out_dir)
        else:
            print(f"  [SKIP] Unknown format")

    print("\nDone.")


if __name__ == "__main__":
    main()
