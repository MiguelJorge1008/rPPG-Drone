"""
RecordPPG.py — Records SW BVP (camera) and HW PPG (Arduino serial) in parallel.

SW: BVP signal per camera frame, mapped via bvp_by_frame (no repeated values).
HW: raw PPG at natural serial rate (~100 Hz), saved separately.

No display — terminal only. Press Ctrl+C to abort early.

Output (in data/):
    BVP_<ALGO>_<ts>_sw.csv   — timestamp, signal_sw  (z-normalised)
    BVP_<ALGO>_<ts>_hw.csv   — timestamp, signal_hw  (raw, relative time)

Usage:
    python RecordPPG.py
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import cv2
import serial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from DataHandler import WebcamHandler, CameraHandler, IMUHandler
from Processor import FaceProcessor
from ROIExtraction import ROI_FOREHEAD, ROI_FACE, ROI_MULTI

XIAO_IP     = "http://192.168.4.1"
SERIAL_PORT = 'COM3'
BAUD_RATE   = 115200
WARMUP_S    = 30
RECORD_S    = 60


def main():
    # --- user input ---
    print("Select video source:  A - Drone   B - Webcam")
    src = input("Option [A/B]: ").strip().upper()
    cam = WebcamHandler(index=0) if src == "B" else CameraHandler(XIAO_IP)
    imu = None if src == "B" else IMUHandler(XIAO_IP)

    print("\nSelect rPPG Algorithm:  1=GREEN  2=OMIT  3=POS_WANG  4=LMS")
    algo = {"1": "GREEN", "2": "OMIT", "3": "POS_WANG", "4": "LMS"}.get(
        input("Option [1-4]: ").strip(), "GREEN"
    )
    if algo == "LMS" and imu is None:
        print("LMS requires drone IMU. Falling back to GREEN.")
        algo = "GREEN"

    print("\nSelect ROI:  1=Forehead  2=Full face  3=Multi-region")
    roi = {1: ROI_FOREHEAD, 2: ROI_FACE, 3: ROI_MULTI}.get(
        int(input("Option [1-3]: ").strip() or "2"), ROI_FACE
    )

    # --- serial (hardware PPG) ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connected to Arduino on {SERIAL_PORT}")
    except Exception:
        ser = None
        print("Warning: Arduino not found. HW signal will not be recorded.")

    hw_rows      = []
    serial_lock  = threading.Lock()
    start_time_ref = [None]   # set once the main loop starts

    def read_serial():
        while ser:
            try:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 3:
                    val = float(parts[1])
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

    proc = FaceProcessor(cam, imu=imu, algo=algo, roi=roi)

    bvp_by_frame = {}
    start_time   = time.time()
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

            if imu is not None:
                sample = imu.get_imu()
                if sample is not None and id(sample) != proc._last_imu_id:
                    proc.imu_signal.append(sample)
                    proc._last_imu_id = id(sample)

            proc.process_frame(frame)
            n = len(proc.rgb_signal)

            if n > 0 and not proc._hr_computing and n % 30 == 0:
                with proc._hr_lock:
                    proc._hr_computing = True
                threading.Thread(target=proc._compute_hr_background, daemon=True).start()

            with proc._hr_lock:
                fresh    = proc._bvps_fresh
                bvp_dict = proc._latest_bvps

            if fresh and bvp_dict is not None and algo in bvp_dict:
                with proc._hr_lock:
                    proc._bvps_fresh = False
                bvp_arr     = bvp_dict[algo]
                n_bvp       = len(bvp_arr)
                start_frame = n - n_bvp
                for i, val in enumerate(bvp_arr):
                    bvp_by_frame[start_frame + i] = val

    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        cam.stop()
        if imu:
            imu.stop()
        if ser:
            ser.close()
        cv2.destroyAllWindows()

        ts_tag = int(time.time())
        out_dir = os.path.join(current_dir, "data")
        os.makedirs(out_dir, exist_ok=True)

        # --- SW CSV (camera frame rate, ~25 fps) ---
        timestamps = list(proc.frame_timestamps)
        df_sw = None
        if timestamps and bvp_by_frame:
            t0 = timestamps[0]
            rows = []
            for fi, ts in enumerate(timestamps):
                if fi not in bvp_by_frame:
                    continue
                rows.append({"timestamp": ts - t0, "signal_sw": bvp_by_frame[fi]})

            if rows:
                df_sw = pd.DataFrame(rows)
                std = df_sw["signal_sw"].std()
                if std > 0:
                    df_sw["signal_sw"] = (df_sw["signal_sw"] - df_sw["signal_sw"].mean()) / std

        if df_sw is not None and not df_sw.empty:
            fname_sw = f"BVP_{algo}_{ts_tag}_sw.csv"
            df_sw.to_csv(os.path.join(out_dir, fname_sw), index=False)
            dur = df_sw["timestamp"].iloc[-1]
            print(f"Saved SW: {len(df_sw)} samples @ ~{len(df_sw)/dur:.1f} Hz → {fname_sw}")
        else:
            print("No SW BVP data in recording window.")
            df_sw = None

        # --- HW CSV (serial rate, ~100 Hz — different fs from SW) ---
        with serial_lock:
            hw_copy = list(hw_rows)

        if hw_copy:
            df_hw = pd.DataFrame(hw_copy)
            df_hw['timestamp'] = df_hw['timestamp'] - df_hw['timestamp'].iloc[0]
            fname_hw = f"BVP_{algo}_{ts_tag}_hw.csv"
            df_hw.to_csv(os.path.join(out_dir, fname_hw), index=False)
            dur_hw = df_hw['timestamp'].iloc[-1]
            print(f"Saved HW: {len(df_hw)} samples @ ~{len(df_hw)/dur_hw:.1f} Hz → {fname_hw}")
        else:
            # no Arduino — save zeros at SW timestamps so file always exists
            if df_sw is not None:
                df_hw_zero = pd.DataFrame({
                    'timestamp': df_sw['timestamp'].values,
                    'signal_hw': np.zeros(len(df_sw))
                })
            else:
                df_hw_zero = pd.DataFrame({'timestamp': [0.0], 'signal_hw': [0.0]})
            fname_hw = f"BVP_{algo}_{ts_tag}_hw.csv"
            df_hw_zero.to_csv(os.path.join(out_dir, fname_hw), index=False)
            print(f"No HW signal — saved zeros → {fname_hw}")


if __name__ == "__main__":
    main()
