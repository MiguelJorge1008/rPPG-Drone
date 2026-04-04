"""
PlotData.py — Plot signals and computed metrics over time for each recording.

Generates one PNG per CSV with 3 rows:
  - Row 0: bandpass-filtered signals (hw + sw) with detected peaks (z-score)
  - Row 1: rate over time (HR_gt vs HR_algo  or  RR_gt vs RR_BARTULA)
  - Row 2: inter-peak intervals over time (IBI in ms  or  BBI in s)

Usage:
    python PlotData.py                          # all recordings (raw)
    python PlotData.py --source clean           # recordings_cleaned/
    python PlotData.py --file Recording_X.csv   # single file
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_RAW      = os.path.join(SCRIPT_DIR, "data")
RECORDINGS_CLEANED  = os.path.join(SCRIPT_DIR, "data_cleaned")
RECORDINGS_LIVE     = os.path.join(SCRIPT_DIR, "recordings")
RESULTS_BASE        = os.path.join(SCRIPT_DIR, "results")
PLOTS_RAW           = os.path.join(RESULTS_BASE, "plots_raw")
PLOTS_CLEAN         = os.path.join(RESULTS_BASE, "plots_clean")
PLOTS_RECORDINGS    = os.path.join(RESULTS_BASE, "plots_recordings")


# ---------------------------------------------------------------------------
# Signal helpers (shared with PlotPeaks)
# ---------------------------------------------------------------------------

def normalise(s):
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
    sig = pd.Series(series).interpolate(limit=5).values
    return sig


def _dedup(sig, t):
    """Keep only indices where the value changes (remove consecutive duplicates)."""
    sig = np.asarray(sig, dtype=float)
    if len(sig) == 0:
        return np.array([], dtype=int)
    idx = np.concatenate([[0], np.where(np.diff(sig) != 0)[0] + 1])
    return idx


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_signals_with_peaks(ax, t, hw, sw, hw_peaks, sw_peaks,
                              hw_filt_n, sw_filt_n,
                              band_label, hw_color, sw_color):
    ax.plot(t, hw_filt_n, color=hw_color, lw=0.8, alpha=0.85, label=f"HW ({band_label}, z-norm)")
    ax.plot(t, sw_filt_n, color=sw_color, lw=0.8, alpha=0.75, label=f"SW ({band_label}, z-norm)")
    if len(hw_peaks):
        ax.scatter(t[hw_peaks], hw_filt_n[hw_peaks],
                   color=hw_color, s=28, zorder=5,
                   label=f"HW peaks ({len(hw_peaks)})")
    if len(sw_peaks):
        ax.scatter(t[sw_peaks], sw_filt_n[sw_peaks],
                   color=sw_color, s=28, zorder=5, marker="^",
                   label=f"SW peaks ({len(sw_peaks)})")
    ax.set_ylabel("Amplitude (z-score)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_ibi(ax, hw_mid, hw_ibi, sw_mid, sw_ibi,
              ylabel, valid_lo, valid_hi,
              hw_color, sw_color):
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

    t  = df["timestamp"].values.astype(float)
    t  = t - t[0]
    dt = np.median(np.diff(t[np.isfinite(t)]))
    fs = 1.0 / dt if dt > 0 else 25.0

    has_hw = "signal_hw" in df.columns and df["signal_hw"].notna().any()

    sw_full = _prep_signal(df["signal_sw"].values, fs)
    sw_idx = _dedup(sw_full, t)
    sw = sw_full[sw_idx]
    t_sw = t[sw_idx]
    fs_sw = 1.0 / np.median(np.diff(t_sw)) if len(t_sw) > 1 else fs

    sw_peaks, sw_filt = _detect_peaks(sw, fs_sw, min_dist_s=0.3)
    sw_filt_n = normalise(sw_filt)
    sw_mid, sw_ibi = _ibi(sw_peaks, t_sw)

    if has_hw:
        hw = _prep_signal(df["signal_hw"].values, fs)
        hw_peaks, hw_filt = _detect_peaks(hw, fs, min_dist_s=0.3)
        hw_filt_n = normalise(hw_filt)
        hw_mid, hw_ibi = _ibi(hw_peaks, t)
    else:
        hw_peaks, hw_filt_n = np.array([], dtype=int), np.array([])
        hw_mid, hw_ibi = np.array([]), np.array([])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(fname, fontsize=11, fontweight="bold")

    # --- Row 0: signals + peaks ---
    ax = axes[0]
    if has_hw:
        ax.plot(t,    hw_filt_n, color="red",   lw=0.8, alpha=0.85, label="HW PPG (z-norm)")
    ax.plot(t_sw, sw_filt_n, color="green", lw=0.8, alpha=0.75, label="SW rPPG (z-norm)")
    if len(hw_peaks):
        ax.scatter(t[hw_peaks],    hw_filt_n[hw_peaks],    color="darkred",   s=28, zorder=5, label=f"HW peaks ({len(hw_peaks)})")
    if len(sw_peaks):
        ax.scatter(t_sw[sw_peaks], sw_filt_n[sw_peaks],    color="darkgreen", s=28, zorder=5, marker="^", label=f"SW peaks ({len(sw_peaks)})")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title("Signals (z-score normalised) with detected peaks")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 1: HR over time ---
    if has_hw and "HR_gt" in df.columns:
        axes[1].plot(t, df["HR_gt"], color="red", lw=1.2, label="HR_gt  (hardware)")
    axes[1].plot(t, df[algo_col], color="green", lw=1.2, label=f"{algo_col}  (camera)", alpha=0.8)
    axes[1].set_ylabel("BPM")
    axes[1].set_title("Heart Rate over time")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # --- Row 2: HRV metrics (SDNN + RMSSD) if available, else IBI ---
    sdnn_algo_col  = f"SDNN_{algo}"
    rmssd_algo_col = f"RMSSD_{algo}"
    has_hrv = (sdnn_algo_col in df.columns or rmssd_algo_col in df.columns)

    ax2 = axes[2]
    if has_hrv:
        if "SDNN_gt" in df.columns and has_hw:
            ax2.plot(t, df["SDNN_gt"],       color="red",        lw=1.0, linestyle="--", label="SDNN_gt")
        if sdnn_algo_col in df.columns:
            ax2.plot(t, df[sdnn_algo_col],   color="green",      lw=1.0, label=f"SDNN_{algo}")
        if "RMSSD_gt" in df.columns and has_hw:
            ax2.plot(t, df["RMSSD_gt"],      color="darkred",    lw=1.0, linestyle="--", label="RMSSD_gt")
        if rmssd_algo_col in df.columns:
            ax2.plot(t, df[rmssd_algo_col],  color="darkgreen",  lw=1.0, linestyle="-.", label=f"RMSSD_{algo}")
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

    _save(fig, fname, out_dir)


def plot_resp(df, fname, out_dir):
    t  = df["timestamp"].values.astype(float)
    t  = t - t[0]
    dt = np.median(np.diff(t[np.isfinite(t)]))
    fs = 1.0 / dt if dt > 0 else 25.0

    hw = _prep_signal(df["signal_hw"].values, fs)
    sw_full = _prep_signal(df["signal_sw"].values, fs)

    sw_idx = _dedup(sw_full, t)
    sw = sw_full[sw_idx]
    t_sw = t[sw_idx]
    fs_sw = 1.0 / np.median(np.diff(t_sw)) if len(t_sw) > 1 else fs

    hw_peaks, hw_filt = _detect_peaks(hw, fs, min_dist_s=1.5)
    sw_peaks, sw_filt = _detect_peaks(sw, fs_sw, min_dist_s=1.5)
    hw_filt_n = normalise(hw_filt)
    sw_filt_n = normalise(sw_filt)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(fname, fontsize=11, fontweight="bold")

    # --- Row 0: signals + peaks (sw uses deduped time axis) ---
    ax = axes[0]
    ax.plot(t,    hw_filt_n, color="blue", lw=0.8, alpha=0.85, label="HW resp (z-norm)")
    ax.plot(t_sw, sw_filt_n, color="cyan", lw=0.8, alpha=0.75, label="SW resp (z-norm)")
    if len(hw_peaks):
        ax.scatter(t[hw_peaks],    hw_filt_n[hw_peaks],    color="darkblue", s=28, zorder=5, label=f"HW peaks ({len(hw_peaks)})")
    if len(sw_peaks):
        ax.scatter(t_sw[sw_peaks], sw_filt_n[sw_peaks],    color="darkcyan", s=28, zorder=5, marker="^", label=f"SW peaks ({len(sw_peaks)})")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title("Signals (z-score normalised) with detected peaks")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Row 1: RR over time ---
    axes[1].plot(t, df["RR_gt"],      color="blue", lw=1.2, label="RR_gt  (hardware)")
    axes[1].plot(t, df["RR_BARTULA"], color="cyan", lw=1.2, label="RR_BARTULA  (camera)", alpha=0.8)
    axes[1].set_ylabel("RPM")
    axes[1].set_title("Respiration Rate over time")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # --- Row 2: BBI ---
    ax = axes[2]
    if "BBI_gt" in df.columns and "BBI_BARTULA" in df.columns:
        # new recordings — use CSV columns directly
        ax.plot(t, df["BBI_gt"],      color="blue", lw=1.2, label="BBI_gt  (hardware)")
        ax.plot(t, df["BBI_BARTULA"], color="cyan", lw=1.2, label="BBI_BARTULA  (camera)", alpha=0.8)
        ax.axhspan(1.5, 10.0, color="gray", alpha=0.08, label="Valid [1.5–10 s]")
        ax.set_ylabel("BBI (s)")
        ax.legend(fontsize=8, loc="upper right")
    else:
        # old recordings — recompute from signal
        hw_mid, hw_bbi = _ibi(hw_peaks, t)
        sw_mid, sw_bbi = _ibi(sw_peaks, t_sw)
        _plot_ibi(ax, hw_mid, hw_bbi, sw_mid, sw_bbi,
                  ylabel="BBI (s)", valid_lo=1.5, valid_hi=10.0,
                  hw_color="blue", sw_color="cyan")
    ax.set_title("Breath-to-Breath Intervals over time")
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
                        choices=["raw", "clean", "recordings"],
                        help="'raw' = data/   'clean' = data_cleaned/   'recordings' = recordings/")
    args = parser.parse_args()

    if args.source == "clean":
        rec_dir = RECORDINGS_CLEANED
        out_dir = PLOTS_CLEAN
    elif args.source == "recordings":
        rec_dir = RECORDINGS_LIVE
        out_dir = PLOTS_RECORDINGS
    else:
        rec_dir = RECORDINGS_RAW
        out_dir = PLOTS_RAW

    if not os.path.isdir(rec_dir):
        print(f"ERROR: {rec_dir} not found.")
        sys.exit(1)

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
            df = pd.read_csv(fpath, on_bad_lines="skip")
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
