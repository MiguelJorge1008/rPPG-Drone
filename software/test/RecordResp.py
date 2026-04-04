"""
RecordResp.py — Records SW respiration (Bartula/camera) and HW respiration
                (Arduino serial belt) in parallel.

SW: Bartula position signal per camera frame, bandpass-filtered after recording.
HW: raw belt signal at natural serial rate (~100 Hz), saved separately.

No display — terminal only. Press Ctrl+C to abort early.

Output (in data/):
    RESP_<ts>_sw.csv  — timestamp, signal_sw  (filtered position signal)
    RESP_<ts>_hw.csv  — timestamp, signal_hw  (raw belt signal, relative time)

Usage:
    python RecordResp.py
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import serial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from DataHandler import WebcamHandler, CameraHandler
from RespiratoryProcessor import RespiratoryProcessor

XIAO_IP     = "http://192.168.4.1"
SERIAL_PORT = 'COM3'
BAUD_RATE   = 115200
WARMUP_S    = 20
RECORD_S    = 60


def main():
    # --- user input ---
    print("Select video source:  A - Drone   B - Webcam")
    src = input("Option [A/B]: ").strip().upper()
    cam = WebcamHandler(index=0) if src == "B" else CameraHandler(XIAO_IP)

    # --- serial (hardware respiration belt) ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to Arduino on {SERIAL_PORT}")
    except Exception:
        ser = None
        print("Warning: Arduino not found. HW signal will not be recorded.")

    hw_rows        = []
    serial_lock    = threading.Lock()
    start_time_ref = [None]

    def read_serial():
        while ser:
            try:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 3:
                    val = float(parts[2])   # respiratory belt is 3rd column
                    ts  = time.time()
                    t0  = start_time_ref[0]
                    if t0 is not None:
                        elapsed = ts - t0
                        if elapsed <= WARMUP_S + RECORD_S:
                            with serial_lock:
                                hw_rows.append({'timestamp': ts, 'signal_hw': val})
            except Exception:
                continue

    if ser:
        threading.Thread(target=read_serial, daemon=True).start()

    proc = RespiratoryProcessor(cam, display=False)
    proc._init_roi_from_pose()

    start_time = time.time()
    start_time_ref[0] = start_time

    try:
        while True:
            elapsed = time.time() - start_time

            if elapsed < WARMUP_S:
                print(f"\rWARMUP: {WARMUP_S - int(elapsed)}s remaining   ", end="", flush=True)
            elif elapsed < WARMUP_S + RECORD_S:
                print(f"\rRECORDING: {WARMUP_S + RECORD_S - int(elapsed)}s remaining   ", end="", flush=True)
            else:
                print("\nRecording complete!")
                break

            frame = cam.get_frame()
            if frame is None:
                continue

            proc.process_frame(frame)

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        cam.stop()
        if ser:
            ser.close()

        ts_tag  = int(time.time())
        out_dir = os.path.join(current_dir, "data")
        os.makedirs(out_dir, exist_ok=True)

        # --- SW CSV (camera frame rate) ---
        timestamps = list(proc.frame_timestamps)
        position   = list(proc.position_signal)

        df_sw = None
        if timestamps and position:
            t0_sw = timestamps[0]
            t_rel = np.array(timestamps) - t0_sw
            fs_sw = 1.0 / np.median(np.diff(t_rel)) if len(t_rel) > 1 else 25.0

            # apply Bartula filter (detrend + bandpass 0.1-0.5 Hz) to full signal
            try:
                sig_filt = RespiratoryProcessor.apply_filters(np.array(position, dtype=float), fs_sw)
            except Exception:
                sig_filt = np.array(position, dtype=float)

            df_sw = pd.DataFrame({'timestamp': t_rel, 'signal_sw': sig_filt})
            fname_sw = f"RESP_{ts_tag}_sw.csv"
            df_sw.to_csv(os.path.join(out_dir, fname_sw), index=False)
            dur = df_sw['timestamp'].iloc[-1]
            print(f"Saved SW: {len(df_sw)} samples @ ~{len(df_sw)/dur:.1f} Hz -> {fname_sw}")
        else:
            print("No SW respiration data.")

        # --- HW CSV (serial rate, ~100 Hz) ---
        with serial_lock:
            hw_copy = list(hw_rows)

        if hw_copy:
            df_hw = pd.DataFrame(hw_copy)
            df_hw['timestamp'] = df_hw['timestamp'] - df_hw['timestamp'].iloc[0]
            fname_hw = f"RESP_{ts_tag}_hw.csv"
            df_hw.to_csv(os.path.join(out_dir, fname_hw), index=False)
            dur_hw = df_hw['timestamp'].iloc[-1]
            print(f"Saved HW: {len(df_hw)} samples @ ~{len(df_hw)/dur_hw:.1f} Hz -> {fname_hw}")
        else:
            if df_sw is not None:
                df_hw_zero = pd.DataFrame({
                    'timestamp': df_sw['timestamp'].values,
                    'signal_hw': np.zeros(len(df_sw))
                })
            else:
                df_hw_zero = pd.DataFrame({'timestamp': [0.0], 'signal_hw': [0.0]})
            fname_hw = f"RESP_{ts_tag}_hw.csv"
            df_hw_zero.to_csv(os.path.join(out_dir, fname_hw), index=False)
            print(f"No HW signal — saved zeros -> {fname_hw}")


if __name__ == "__main__":
    main()
