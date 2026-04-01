import serial
import threading
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os

# --- PATH MAGIC ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

# --- CORRECTED IMPORTS ---
from DataHandler import WebcamHandler, CameraHandler, IMUHandler
from Processor import FaceProcessor
from ROIExtraction import ROI_FOREHEAD, ROI_FACE, ROI_MULTI
from RespiratoryProcessor import RespiratoryProcessor

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  
BAUD_RATE = 115200
DISPLAY_SECS   = 20    # time window shown in both plots (seconds)
RR_WINDOW_S    = 30    # GT computation window (matches RespiratoryProcessor)
WINDOW_SIZE_HW = 2000  # max hardware samples kept in memory
WINDOW_SIZE_SW = 600   # max camera samples kept in memory
BUFFER_UPDATE  = 5
XIAO_IP = "http://192.168.4.1"

class RecordRespPlotter:
    def __init__(self, port, baud, cam, cam_choice, display=True):
        self.hw_time = []
        self.hw_resp = []
        self.csv_rows = []

        self.cam_choice = cam_choice
        self.display = display
        self.serial_lock = threading.Lock()

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"Connected to Arduino on {port}")
        except:
            self.ser = None
            print("Warning: Arduino not found. Hardware plot will be empty.")

        self.cam = cam
        self.proc = RespiratoryProcessor(self.cam)

        if self.display:
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
            self.fig.suptitle('Data Collection: Hardware Respiration vs Camera Estimation', fontsize=16)

            self.line_hw, = self.ax1.plot([], [], 'b-', label='Hardware Breathing', lw=2)
            self.line_sw, = self.ax2.plot([], [], 'c-', label='Camera Breathing', lw=2)

            self.ax1.set_title("Arduino Ground Truth (Live Stream)")
            self.ax2.set_title("Camera Estimation (Chest Displacement)")

            for ax in [self.ax1, self.ax2]:
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")
                ax.set_xlabel("Time (s)")

            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
            self.txt_hw = self.ax1.text(0.02, 0.90, 'Gathering Data...', transform=self.ax1.transAxes,
                                        fontsize=14, fontweight='bold', color='darkblue', zorder=10, bbox=bbox_props)
            self.txt_sw = self.ax2.text(0.02, 0.90, 'Gathering Data...', transform=self.ax2.transAxes,
                                        fontsize=14, fontweight='bold', color='teal', zorder=10, bbox=bbox_props)

            self.phase_text = self.fig.text(0.5, 0.05, 'INITIALIZING...', ha='center', va='center',
                                            fontsize=16, fontweight='bold', color='white',
                                            bbox=dict(facecolor='black', pad=5, alpha=0.9))

        self.running = True
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)

    def read_serial(self):
        while self.running and self.ser:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        r_volts = float(parts[2])  
                        with self.serial_lock:
                            self.hw_time.append(time.time())
                            self.hw_resp.append(r_volts)
            except:
                continue

    def run(self):
        self.proc._init_roi_from_pose()
        
        self.serial_thread.start()
        frame_count = 0
        self.start_time = time.time()
        
        current_hw_rpm = np.nan
        current_sw_rpm = np.nan
        current_hw_brv = np.nan
        current_sw_brv = np.nan
        filt_sw = []
        
        try:
            while self.running:
                elapsed = time.time() - self.start_time
                if elapsed < 30:
                    if self.display:
                        self.phase_text.set_text(f'WARMUP: {30 - int(elapsed)}s remaining (No data saved)')
                        self.phase_text.get_bbox_patch().set_facecolor('orange')
                    else:
                        print(f'\rWARMUP: {30 - int(elapsed)}s remaining (No data saved)', end='', flush=True)
                elif elapsed < 90:
                    if self.display:
                        self.phase_text.set_text(f'RECORDING: {90 - int(elapsed)}s remaining')
                        self.phase_text.get_bbox_patch().set_facecolor('red')
                    else:
                        print(f'\rRECORDING: {90 - int(elapsed)}s remaining', end='', flush=True)
                else:
                    print("\nRecording complete! Generating CSV...")
                    self.running = False
                    break

                frame = self.cam.get_frame()
                if frame is None: continue
                
                annotated = self.proc.process_frame(frame)
                n_frames = len(self.proc.position_signal)
                
                if n_frames > 0 and not self.proc._rr_computing and n_frames % 30 == 0:
                    with self.proc._rr_lock:
                        self.proc._rr_computing = True
                    threading.Thread(target=self.proc._compute_rr_background, daemon=True).start()

                frame_count += 1
                if frame_count % BUFFER_UPDATE == 0:
                    
                    with self.serial_lock:
                        hw_t_list = list(self.hw_time)[-WINDOW_SIZE_HW:]
                        hw_r_list = list(self.hw_resp)[-WINDOW_SIZE_HW:]
                    
                    min_len = min(len(hw_t_list), len(hw_r_list))
                    
                    # 1. HARDWARE — filter to last DISPLAY_SECS, use time x-axis
                    if min_len > 50:
                        hw_t_arr = np.array(hw_t_list[:min_len])
                        hw_r_arr = np.array(hw_r_list[:min_len])

                        # keep only last DISPLAY_SECS seconds for display
                        t_cutoff = hw_t_arr[-1] - DISPLAY_SECS
                        mask = hw_t_arr >= t_cutoff
                        hw_t_disp = hw_t_arr[mask] - hw_t_arr[mask][0] if mask.any() else hw_t_arr
                        hw_r_disp = hw_r_arr[mask]

                        if self.display:
                            self.line_hw.set_xdata(hw_t_disp)
                            self.line_hw.set_ydata(hw_r_disp)
                            self.ax1.set_xlim(0, DISPLAY_SECS)
                            y_min, y_max = np.min(hw_r_disp), np.max(hw_r_disp)
                            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                            self.ax1.set_ylim(y_min - margin, y_max + margin)

                        # RR via find_peaks — identical to RespiratoryProcessor._estimate_rr_from_peaks
                        # use last RR_WINDOW_S seconds for GT computation
                        t_cutoff_rr = hw_t_arr[-1] - RR_WINDOW_S
                        mask_rr = hw_t_arr >= t_cutoff_rr
                        hw_t_calc = hw_t_arr[mask_rr]
                        hw_r_calc = hw_r_arr[mask_rr]
                        dt_hw_rr = np.median(np.diff(hw_t_calc)) if len(hw_t_calc) > 1 else 0
                        if dt_hw_rr > 0 and len(hw_r_calc) >= 10:
                            fs_hw_rr = 1.0 / dt_hw_rr
                            min_dist = max(int(1.5 * fs_hw_rr), 1)
                            hw_peaks, _ = find_peaks(hw_r_calc, distance=min_dist,
                                                     prominence=np.std(hw_r_calc) * 0.25)
                            if len(hw_peaks) >= 2:
                                bb_intervals = [
                                    (hw_t_calc[hw_peaks[i+1]] - hw_t_calc[hw_peaks[i]])
                                    for i in range(len(hw_peaks) - 1)
                                    if 1.5 <= (hw_t_calc[hw_peaks[i+1]] - hw_t_calc[hw_peaks[i]]) <= 10.0
                                ]
                                if len(bb_intervals) >= 2:
                                    rr_val = 60.0 / np.median(bb_intervals)
                                    if 6.0 <= rr_val <= 60.0:
                                        current_hw_rpm = rr_val
                                        current_hw_brv = np.std(bb_intervals) * 1000
                                        if self.display:
                                            self.txt_hw.set_text(f"RPM: {current_hw_rpm:.1f} | BRV: {current_hw_brv:.0f} ms")

                    # 2. SOFTWARE — use time x-axis aligned to DISPLAY_SECS
                    if n_frames > 64:
                        fs = self.proc.get_fps()
                        sw_n = int(DISPLAY_SECS * fs)
                        raw_sw = np.array(self.proc.position_signal[-sw_n:])

                        try:
                            filt_sw = self.proc.apply_filters(raw_sw, fs)
                            sw_t = np.arange(len(filt_sw)) / fs

                            if self.display:
                                self.line_sw.set_xdata(sw_t)
                                self.line_sw.set_ydata(filt_sw)
                                self.ax2.set_xlim(0, DISPLAY_SECS)
                                y_min, y_max = np.min(filt_sw), np.max(filt_sw)
                                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                                self.ax2.set_ylim(y_min - margin, y_max + margin)

                            sw_peaks, _ = find_peaks(filt_sw, distance=int(1.5 * fs), prominence=np.std(filt_sw) * 0.25)
                            if len(sw_peaks) >= 2:
                                bb_sw = np.diff(sw_peaks) / fs
                                current_sw_brv = np.std(bb_sw) * 1000

                            if self.proc.rr_estimate is not None:
                                current_sw_rpm = self.proc.rr_estimate
                                if self.display:
                                    self.txt_sw.set_text(f"RPM: {current_sw_rpm:.1f} | BRV: {current_sw_brv:.0f} ms")

                        except Exception:
                            pass

                    if self.display:
                        self.fig.canvas.draw()
                        self.fig.canvas.flush_events()
                    
                # --- LIVE RECORDING LOGIC ---
                if 30 <= elapsed <= 90:
                    t_cam = self.proc.frame_timestamps[-1] if self.proc.frame_timestamps else time.time()
                    
                    hw_sig = np.nan
                    with self.serial_lock:
                        if len(self.hw_resp) > 0:
                            hw_sig = self.hw_resp[-1]
                            
                    sw_sig = np.nan
                    if len(filt_sw) > 0:
                        sw_sig = filt_sw[-1] 
                        
                    self.csv_rows.append({
                        'timestamp': t_cam,
                        'signal_hw': hw_sig,
                        'signal_sw': sw_sig,
                        'RR_gt': current_hw_rpm,
                        'RR_BARTULA': current_sw_rpm,
                        'BRV_gt': current_hw_brv,
                        'BRV_BARTULA': current_sw_brv,
                        'source': "webcam" if self.cam_choice == "B" else "drone"
                    })

                if self.display:
                    cv2.imshow("Recording Feed (Chest ROI)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Recording manually aborted!")
                    break

        finally:
            self.cam.stop()
            cv2.destroyAllWindows()
            if self.ser: self.ser.close()

            if len(self.csv_rows) > 0:
                df = pd.DataFrame(self.csv_rows)
                filename = f"Recording_Resp_Bartula_{int(time.time())}.csv"
                recordings_dir = os.path.join(current_dir, "recordings")
                os.makedirs(recordings_dir, exist_ok=True)
                filepath = os.path.join(recordings_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved {len(df)} live frames to {filepath}")


if __name__ == "__main__":
    print("--- Respiration Data Collection System ---")

    print("\nShow graphs and camera window?")
    print("  1 - Yes (graphs + camera)")
    print("  2 - No  (terminal only)")
    display = input("Option [1/2]: ").strip() != "2"

    print("\nSelect video source:")
    print("  A - Drone camera (XIAO ESP32)")
    print("  B - PC webcam")
    src_choice = input("Option [A/B]: ").strip().upper()

    if src_choice == "B":
        cam = WebcamHandler(index=0)
    else:
        cam = CameraHandler(XIAO_IP)

    print("\nWaiting for camera to initialize...")
    print("WARNING: Please stand/sit so the camera can clearly see your shoulders and chest!")
    print("Get ready! The 30-second warmup will start automatically after posture lock.")
    
    plotter = RecordRespPlotter(SERIAL_PORT, BAUD_RATE, cam, src_choice, display=display)
    plotter.run()