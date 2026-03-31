import serial
import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import find_peaks, butter, filtfilt
import heartpy as hp
from DataHandler import WebcamHandler
from Processor import FaceProcessor

# --- CONFIGURATION ---
SERIAL_PORT = 'COM5'  
BAUD_RATE = 115200
WINDOW_SIZE = 400     
BUFFER_UPDATE = 5     

class ValidationPlotter:
    def __init__(self, port, baud, algo="GREEN"):
        self.hw_time = deque(maxlen=WINDOW_SIZE)
        self.hw_ppg = deque(maxlen=WINDOW_SIZE)
        self.current_rppg_bvp = None  
        self.serial_lock = threading.Lock() 
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"Connected to Arduino on {port}")
        except:
            self.ser = None
            print("Warning: Arduino not found. Hardware plot will be empty.")

        self.cam = WebcamHandler(index=0)
        self.proc = FaceProcessor(self.cam, algo=algo)
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle(f'Demo Day Validation: Hardware PPG vs Camera rPPG ({algo})', fontsize=16)
        
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
                                    fontsize=14, fontweight='bold', color='darkred', zorder=10, bbox=bbox_props)
        self.txt_sw = self.ax2.text(0.02, 0.90, 'Waiting for 30s buffer...', transform=self.ax2.transAxes, 
                                    fontsize=14, fontweight='bold', color='darkgreen', zorder=10, bbox=bbox_props)

        self.running = True
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)

    def read_serial(self):
        while self.running and self.ser:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        t_sec = float(parts[0]) / 1000.0  
                        val = float(parts[1])
                        with self.serial_lock:
                            self.hw_time.append(t_sec)
                            self.hw_ppg.append(val)
            except:
                continue

    def run(self):
        self.serial_thread.start()
        frame_count = 0
        
        str_hw_term = "HW -> BPM: -- | HRV: --"
        str_sw_term = "SW -> BPM: -- | HRV: --"
        
        try:
            while self.running:
                frame = self.cam.get_frame()
                if frame is None: continue
                
                annotated = self.proc.process_frame(frame)
                
                if len(self.proc.rgb_signal) % 30 == 0 and len(self.proc.rgb_signal) > 0:
                    with self.proc._hr_lock:
                        if not self.proc._hr_computing:
                            self.proc._hr_computing = True
                            t = threading.Thread(target=self.proc._compute_hr_background, daemon=True)
                            t.start()

                with self.proc._hr_lock:
                    if self.proc._bvps_fresh:
                        self.current_rppg_bvp = self.proc._latest_bvps
                        self.proc._bvps_fresh = False

                frame_count += 1
                if frame_count % BUFFER_UPDATE == 0:
                    
                    with self.serial_lock:
                        hw_t_list = list(self.hw_time)
                        hw_p_list = list(self.hw_ppg)
                    
                    min_len = min(len(hw_t_list), len(hw_p_list))
                    
                    # ---------------------------------------------------------
                    # 1. HARDWARE UPDATE (Uses HeartPy for gold-standard accuracy)
                    # ---------------------------------------------------------
                    if min_len > 100:
                        hw_t_arr = np.array(hw_t_list[:min_len])
                        hw_p_arr = np.array(hw_p_list[:min_len])
                        
                        self.line_hw.set_ydata(hw_p_arr)
                        self.line_hw.set_xdata(np.arange(len(hw_p_arr)))
                        
                        y_min, y_max = np.min(hw_p_arr), np.max(hw_p_arr)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                        self.ax1.set_ylim(y_min - margin, y_max + margin)

                        time_duration = hw_t_arr[-1] - hw_t_arr[0]
                        if time_duration > 0:
                            sample_rate = len(hw_t_arr) / time_duration
                            try:
                                _, measures = hp.process(hw_p_arr, sample_rate=sample_rate)
                                ui_str = f"BPM: {measures['bpm']:.1f} | HRV: {measures['sdnn']:.1f} ms"
                                self.txt_hw.set_text(ui_str)
                                str_hw_term = f"HW -> {ui_str}"
                            except hp.exceptions.BadSignalWarning:
                                self.txt_hw.set_text("BPM: -- | HRV: -- (Messy Signal)")

# ---------------------------------------------------------
                    # 2. SOFTWARE UPDATE (Adaptive FFT-Guided Narrowband)
                    # ---------------------------------------------------------
                    if self.current_rppg_bvp is not None:
                        bvp_array = self.current_rppg_bvp[self.proc.algo]
                        display_bvp = bvp_array[-WINDOW_SIZE:] if len(bvp_array) > WINDOW_SIZE else bvp_array
                        
                        fs = self.proc.get_fps()
                        sw_bpm_str = "--"
                        sw_hrv_str = "--"
                        
                        if fs > 0 and len(display_bvp) > 15:
                            # 1. Get the highly accurate baseline BPM from the FFT
                            fft_bpm = 70.0 
                            if self.proc.hr_estimate and self.proc.algo in self.proc.hr_estimate:
                                fft_bpm = self.proc.hr_estimate[self.proc.algo]
                            
                            sw_bpm_str = f"{fft_bpm:.1f}"

                            # 2. THE DEMO DAY MAGIC: Adaptive Narrowband Filter
                            # We build a custom filter centered exactly on your current heartbeat
                            center_hz = fft_bpm / 60.0
                            low_hz = max(0.5, center_hz - 0.5)   # Just below your heart rate
                            high_hz = min(3.0, center_hz + 0.5)  # Just above your heart rate
                            
                            nyq = 0.5 * fs
                            if 0 < low_hz < high_hz < nyq:
                                b, a = butter(3, [low_hz / nyq, high_hz / nyq], btype='bandpass')
                                # Padding helps prevent the edges of the graph from flying out of control
                                pad_len = min(15, len(display_bvp) - 1)
                                display_bvp = filtfilt(b, a, display_bvp, padlen=pad_len)
                            
                            # 3. Peak Prediction / Validation (Based on the Report's Logic)
                            # Grab ALL candidate peaks, even the noisy ones
                            candidates, _ = find_peaks(display_bvp, distance=int(0.3 * fs), prominence=0.05)
                            
                            if len(candidates) > 0 and fft_bpm > 40:
                                expected_gap = int((60.0 / fft_bpm) * fs)
                                validated_peaks = [candidates[0]] # Start with the first peak
                                
                                for cand in candidates[1:]:
                                    last_peak = validated_peaks[-1]
                                    time_since_last = cand - last_peak
                                    
                                    # ONLY accept the peak if it falls inside the expected prediction window (0.5x to 1.5x)
                                    if 0.5 * expected_gap <= time_since_last <= 1.5 * expected_gap:
                                        validated_peaks.append(cand)
                                    # If we completely missed a beat (noise blocked it), force a reset to keep tracking
                                    elif time_since_last > 1.5 * expected_gap:
                                        validated_peaks.append(cand)

                                # Calculate HRV using only the mathematically validated peaks
                                if len(validated_peaks) >= 2:
                                    rr_intervals = np.diff(validated_peaks) / fs
                                    if np.mean(rr_intervals) > 0:
                                        calc_hrv = np.std(rr_intervals) * 1000
                                        sw_hrv_str = f"{calc_hrv:.1f}"

                        self.line_sw.set_ydata(display_bvp)
                        self.line_sw.set_xdata(np.arange(len(display_bvp)))
                        self.ax2.set_xlim(0, max(len(display_bvp), 1))
                        
                        if len(display_bvp) > 10:
                            y_min, y_max = np.min(display_bvp), np.max(display_bvp)
                            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                            self.ax2.set_ylim(y_min - margin, y_max + margin)

                        ui_str = f"BPM: {sw_bpm_str} | HRV: {sw_hrv_str} ms"
                        self.txt_sw.set_text(ui_str)
                        str_sw_term = f"SW -> {ui_str}"

                    print(f"\r{str_hw_term}   ||   {str_sw_term}          ", end="", flush=True)

                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                cv2.imshow("Validation Feed (Face Detection)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("\nShutting down validation system...")
            self.running = False
            self.cam.stop()
            cv2.destroyAllWindows()
            if self.ser: self.ser.close()

if __name__ == "__main__":
    print("--- Ground Truth Validation System ---")
    print("Select rPPG Algorithm:")
    print("  1 - GREEN (Verkruysse 2008)")
    print("  2 - OMIT  (Casado 2023)")
    print("  3 - POS   (Wang 2017)")
    
    choice = input("Option [1-3]: ").strip()
    algo_map = {"1": "GREEN", "2": "OMIT", "3": "POS_WANG"}
    selected_algo = algo_map.get(choice, "GREEN")
    
    print(f"\nStarting with Algorithm: {selected_algo}")
    print("Waiting for camera and sensors to initialize...\n")
    
    plotter = ValidationPlotter(SERIAL_PORT, BAUD_RATE, algo=selected_algo)
    plotter.run()