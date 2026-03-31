import serial
import threading
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from DataHandler import WebcamHandler, CameraHandler, IMUHandler
from Processor import FaceProcessor
from ROIExtraction import ROI_FOREHEAD, ROI_FACE, ROI_MULTI

# --- CONFIGURATION ---
SERIAL_PORT = 'COM5'  
BAUD_RATE = 115200
WINDOW_SIZE = 400     
BUFFER_UPDATE = 5     
XIAO_IP = "http://192.168.4.1"

class RecordPlotter:
    def __init__(self, port, baud, cam, cam_choice, imu=None, algo="GREEN", roi=ROI_FACE):
        self.hw_time = [] 
        self.hw_ppg = []
        self.csv_rows = [] 
        
        self.cam_choice = cam_choice
        self.serial_lock = threading.Lock() 
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"Connected to Arduino on {port}")
        except:
            self.ser = None
            print("Warning: Arduino not found. Hardware plot will be empty.")

        self.cam = cam
        self.proc = FaceProcessor(self.cam, imu=imu, algo=algo, roi=roi)
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle(f'Data Collection: Hardware PPG vs Camera rPPG ({algo} | {roi})', fontsize=16)
        
        self.line_hw, = self.ax1.plot([], [], 'r-', label='Hardware PPG', lw=2)
        self.line_sw, = self.ax2.plot([], [], 'g-', label=f'rPPG ({algo})', lw=2)
        
        self.ax1.set_title("Arduino Ground Truth (Live Stream)")
        self.ax2.set_title("Camera Estimation (Block Updated)")
        
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(0, WINDOW_SIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
        self.txt_hw = self.ax1.text(0.02, 0.90, 'Gathering Data...', transform=self.ax1.transAxes, 
                                    fontsize=12, fontweight='bold', color='darkred', zorder=10, bbox=bbox_props)
        self.txt_sw = self.ax2.text(0.02, 0.90, 'Waiting for buffer...', transform=self.ax2.transAxes, 
                                    fontsize=12, fontweight='bold', color='darkgreen', zorder=10, bbox=bbox_props)

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
                        val = float(parts[1])
                        with self.serial_lock:
                            self.hw_time.append(time.time())
                            self.hw_ppg.append(val)
            except:
                continue

    def run(self):
        self.serial_thread.start()
        frame_count = 0
        self.start_time = time.time()
        
        current_hw_bpm = np.nan
        current_sw_bpm = np.nan
        current_hw_sdnn = np.nan
        current_sw_sdnn = np.nan
        current_hw_rmssd = np.nan
        current_sw_rmssd = np.nan
        
        display_bvp = []
        
        try:
            while self.running:
                elapsed = time.time() - self.start_time
                if elapsed < 60:
                    self.phase_text.set_text(f'WARMUP: {60 - int(elapsed)}s remaining (No data saved)')
                    self.phase_text.get_bbox_patch().set_facecolor('orange')
                elif elapsed < 120:
                    self.phase_text.set_text(f'🔴 RECORDING: {120 - int(elapsed)}s remaining')
                    self.phase_text.get_bbox_patch().set_facecolor('red')
                else:
                    print("\nRecording complete! Generating CSV...")
                    self.running = False
                    break

                frame = self.cam.get_frame()
                if frame is None: continue
                
                annotated = self.proc.process_frame(frame)
                
                if len(self.proc.rgb_signal) % 30 == 0 and len(self.proc.rgb_signal) > 0:
                    with self.proc._hr_lock:
                        if not self.proc._hr_computing:
                            self.proc._hr_computing = True
                            t = threading.Thread(target=self.proc._compute_hr_background, daemon=True)
                            t.start()

                frame_count += 1
                if frame_count % BUFFER_UPDATE == 0:
                    
                    with self.serial_lock:
                        hw_t_list = list(self.hw_time)[-WINDOW_SIZE:] 
                        hw_p_list = list(self.hw_ppg)[-WINDOW_SIZE:]
                    
                    min_len = min(len(hw_t_list), len(hw_p_list))
                    
                    # 1. HARDWARE UPDATE 
                    if min_len > 100:
                        hw_t_arr = np.array(hw_t_list[:min_len])
                        hw_p_arr = np.array(hw_p_list[:min_len])
                        
                        self.line_hw.set_ydata(hw_p_arr)
                        self.line_hw.set_xdata(np.arange(len(hw_p_arr)))
                        y_min, y_max = np.min(hw_p_arr), np.max(hw_p_arr)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                        self.ax1.set_ylim(y_min - margin, y_max + margin)

                        hw_peaks, _ = find_peaks(hw_p_arr, distance=15, prominence=0.05)
                        if len(hw_peaks) >= 3: # Need at least 3 peaks to get 2 intervals for RMSSD
                            rr_intervals = np.diff(hw_t_arr[hw_peaks])
                            rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 1.5)]

                            if len(rr_intervals) >= 2:
                                median_rr = np.median(rr_intervals)
                                mad = np.median(np.abs(rr_intervals - median_rr))
                                rr_intervals = rr_intervals[np.abs(rr_intervals - median_rr) < 3 * mad]
                                
                                if len(rr_intervals) >= 2:
                                    current_hw_bpm = 60.0 / np.mean(rr_intervals)
                                    current_hw_sdnn = np.std(rr_intervals) * 1000
                                    # RMSSD Math
                                    current_hw_rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000
                                    self.txt_hw.set_text(f"BPM: {current_hw_bpm:.1f} | SDNN: {current_hw_sdnn:.0f} | RMSSD: {current_hw_rmssd:.0f}")

                    # 2. SOFTWARE UPDATE 
                    with self.proc._hr_lock:
                        bvp_dict = self.proc._latest_bvps
                        
                    if bvp_dict is not None and self.proc.algo in bvp_dict:
                        bvp_array = bvp_dict[self.proc.algo]
                        display_bvp = bvp_array[-WINDOW_SIZE:] if len(bvp_array) > WINDOW_SIZE else bvp_array
                        
                        if len(display_bvp) > 15:
                            fs = self.proc.get_fps()
                            if self.proc.hr_estimate and self.proc.algo in self.proc.hr_estimate:
                                current_sw_bpm = self.proc.hr_estimate[self.proc.algo]
                                
                            bvp_std = np.std(display_bvp)
                            if bvp_std > 0:
                                display_bvp = (display_bvp - np.mean(display_bvp)) / bvp_std
                                
                            # Re-add broad peak finder for live Software HRV & RMSSD
                            dist = max(1, int(0.3 * fs))
                            sw_peaks, _ = find_peaks(display_bvp, distance=dist, prominence=0.1)
                            if len(sw_peaks) >= 3:
                                rr_sw = np.diff(sw_peaks) / fs
                                rr_sw = rr_sw[(rr_sw > 0.3) & (rr_sw < 1.5)]
                                if len(rr_sw) >= 2:
                                    current_sw_sdnn = np.std(rr_sw) * 1000
                                    current_sw_rmssd = np.sqrt(np.mean(np.diff(rr_sw)**2)) * 1000
                                    
                            self.txt_sw.set_text(f"BPM: {current_sw_bpm:.1f} | SDNN: {current_sw_sdnn:.0f} | RMSSD: {current_sw_rmssd:.0f}")

                        self.line_sw.set_ydata(display_bvp)
                        self.line_sw.set_xdata(np.arange(len(display_bvp)))
                        self.ax2.set_xlim(0, max(len(display_bvp), 1))
                        
                        if len(display_bvp) > 10:
                            y_min, y_max = np.min(display_bvp), np.max(display_bvp)
                            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                            self.ax2.set_ylim(y_min - margin, y_max + margin)

                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                # --- LIVE RECORDING LOGIC ---
                if 60 <= elapsed <= 120:
                    t_cam = self.proc.frame_timestamps[-1] if self.proc.frame_timestamps else time.time()
                    
                    hw_sig = np.nan
                    with self.serial_lock:
                        if len(self.hw_ppg) > 0:
                            hw_sig = self.hw_ppg[-1]
                            
                    sw_sig = np.nan
                    if len(display_bvp) > 0:
                        sw_sig = display_bvp[-1] 
                        
                    self.csv_rows.append({
                        'timestamp': t_cam,
                        'signal_hw': hw_sig,
                        'signal_sw': sw_sig,
                        'HR_gt': current_hw_bpm,
                        f'HR_{self.proc.algo}': current_sw_bpm,
                        'SDNN_gt': current_hw_sdnn,
                        f'SDNN_{self.proc.algo}': current_sw_sdnn,
                        'RMSSD_gt': current_hw_rmssd,
                        f'RMSSD_{self.proc.algo}': current_sw_rmssd,
                        'roi_mode': self.proc.roi,
                        'source': "webcam" if self.cam_choice == "B" else "drone"
                    })

                cv2.imshow("Recording Feed (Face Detection)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Recording manually aborted!")
                    break

        finally:
            self.cam.stop()
            if hasattr(self.proc, 'imu') and self.proc.imu:
                self.proc.imu.stop()
            cv2.destroyAllWindows()
            if self.ser: self.ser.close()
            
            if len(self.csv_rows) > 0:
                df = pd.DataFrame(self.csv_rows)
                filename = f"Recording_rPPG_{self.proc.algo}_{int(time.time())}.csv"
                df.to_csv(f"recordings/{filename}", index=False)
                print(f"✅ Success! Saved {len(df)} live frames to {filename}")


if __name__ == "__main__":
    print("--- rPPG Data Collection System ---")
    
    print("\nSelect video source:")
    print("  A - Drone camera (XIAO ESP32)")
    print("  B - PC webcam")
    src_choice = input("Option [A/B]: ").strip().upper()

    if src_choice == "B":
        cam = WebcamHandler(index=0)
        imu = None
    else:
        cam = CameraHandler(XIAO_IP)
        imu = IMUHandler(XIAO_IP)

    print("\nSelect rPPG Algorithm:")
    print("  1 - GREEN (Verkruysse 2008)")
    print("  2 - OMIT  (Casado 2023)")
    print("  3 - POS   (Wang 2017)")
    print("  4 - LMS   (adaptive; requires IMU/drone)")
    
    algo_choice = input("Option [1-4]: ").strip()
    algo_map = {"1": "GREEN", "2": "OMIT", "3": "POS_WANG", "4": "LMS"}
    selected_algo = algo_map.get(algo_choice, "GREEN")
    
    if selected_algo == "LMS" and imu is None:
        print("Warning: LMS requires drone IMU. Falling back to GREEN.")
        selected_algo = "GREEN"

    print("\nSelect ROI:")
    print("  1 - Forehead       (4 landmarks — original)")
    print("  2 - Full face      (36 landmarks — face oval)")
    print("  3 - Multi-region   (face oval + 9x9 grid + DMRS; Face2PPG)")
    roi_choice = input("Option [1-3]: ").strip()
    roi_map = {"1": ROI_FOREHEAD, "2": ROI_FACE, "3": ROI_MULTI}
    selected_roi = roi_map.get(roi_choice, ROI_FACE)

    print(f"\nStarting with Algorithm: {selected_algo} | ROI: {selected_roi}")
    print("Get ready! The 60-second warmup will start automatically.")
    
    plotter = RecordPlotter(SERIAL_PORT, BAUD_RATE, cam, src_choice, imu, algo=selected_algo, roi=selected_roi)
    plotter.run()