import serial
import threading
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import find_peaks
from DataHandler import WebcamHandler, CameraHandler
from RespiratoryProcessor import RespiratoryProcessor

# --- CONFIGURATION ---
SERIAL_PORT = 'COM5'  
BAUD_RATE = 115200
WINDOW_SIZE_HW = 1000  
WINDOW_SIZE_SW = 600   
BUFFER_UPDATE = 5      
XIAO_IP = "http://192.168.4.1"

class ValidationRespPlotter:
    def __init__(self, port, baud, cam):
        self.hw_time = deque(maxlen=WINDOW_SIZE_HW)
        self.hw_resp = deque(maxlen=WINDOW_SIZE_HW)
        self.serial_lock = threading.Lock() 
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            print(f"Connected to Arduino on {port}")
        except:
            self.ser = None
            print("Warning: Arduino not found. Hardware plot will be empty.")

        self.cam = cam
        self.proc = RespiratoryProcessor(self.cam)
        
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle('Demo Day Validation: Hardware Respiration vs Camera Estimation', fontsize=16)
        
        self.line_hw, = self.ax1.plot([], [], 'b-', label='Hardware Breathing (Piezo/Stretch)', lw=2)
        self.line_sw, = self.ax2.plot([], [], 'c-', label='Camera Breathing (Bartula 2013)', lw=2)
        
        self.ax1.set_title("Arduino Ground Truth (Live Stream)")
        self.ax2.set_title("Camera Estimation (Chest Displacement)")
        
        for ax in [self.ax1, self.ax2]:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
        self.txt_hw = self.ax1.text(0.02, 0.90, 'Gathering Data...', transform=self.ax1.transAxes, 
                                    fontsize=14, fontweight='bold', color='darkblue', zorder=10, bbox=bbox_props)
        self.txt_sw = self.ax2.text(0.02, 0.90, 'Gathering Data...', transform=self.ax2.transAxes, 
                                    fontsize=14, fontweight='bold', color='teal', zorder=10, bbox=bbox_props)

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
                        r_volts = float(parts[2])  
                        
                        with self.serial_lock:
                            self.hw_time.append(t_sec)
                            self.hw_resp.append(r_volts)
            except:
                continue

    def run(self):
        self.proc._init_roi_from_pose()
        
        self.serial_thread.start()
        frame_count = 0
        
        str_hw_term = "HW -> RPM: --"
        str_sw_term = "SW -> RPM: --"
        
        try:
            while self.running:
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
                        hw_t_list = list(self.hw_time)
                        hw_r_list = list(self.hw_resp)
                    
                    min_len = min(len(hw_t_list), len(hw_r_list))
                    
                    if min_len > 50:
                        hw_t_arr = np.array(hw_t_list[:min_len])
                        hw_r_arr = np.array(hw_r_list[:min_len])
                        
                        self.line_hw.set_ydata(hw_r_arr)
                        self.line_hw.set_xdata(np.arange(len(hw_r_arr)))
                        
                        self.ax1.set_xlim(0, WINDOW_SIZE_HW)
                        y_min, y_max = np.min(hw_r_arr), np.max(hw_r_arr)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                        self.ax1.set_ylim(y_min - margin, y_max + margin)

                        hw_peaks, _ = find_peaks(hw_r_arr, distance=75, prominence=0.05)
                        
                        if len(hw_peaks) >= 2:
                            bb_intervals = np.diff(hw_t_arr[hw_peaks])
                            if np.mean(bb_intervals) > 0:
                                resp_rate = 60.0 / np.mean(bb_intervals)
                                ui_str = f"RPM: {resp_rate:.1f} breaths/min"
                                self.txt_hw.set_text(ui_str)
                                str_hw_term = f"HW -> {ui_str}"
                        else:
                            self.txt_hw.set_text("RPM: --")

                    if n_frames > 64:
                        raw_sw = np.array(self.proc.position_signal[-WINDOW_SIZE_SW:])
                        fs = self.proc.get_fps()
                        
                        try:
                            filt_sw = self.proc.apply_filters(raw_sw, fs)
                            
                            self.line_sw.set_ydata(filt_sw)
                            self.line_sw.set_xdata(np.arange(len(filt_sw)))
                            
                            self.ax2.set_xlim(0, max(len(filt_sw), 1))
                            y_min, y_max = np.min(filt_sw), np.max(filt_sw)
                            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                            self.ax2.set_ylim(y_min - margin, y_max + margin)
                            
                            sw_rpm_str = "--"
                            if self.proc.rr_estimate is not None:
                                sw_rpm_str = f"{self.proc.rr_estimate:.1f}"
                                
                            motion_now = self.proc.motion_flags[-1] if self.proc.motion_flags else False
                            status = " [MOTION DETECTED]" if motion_now else ""
                            
                            ui_str = f"RPM: {sw_rpm_str} breaths/min{status}"
                            self.txt_sw.set_text(ui_str)
                            str_sw_term = f"SW -> {ui_str}"
                            
                        except Exception:
                            pass 

                    print(f"\r{str_hw_term}   ||   {str_sw_term}          ", end="", flush=True)

                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

                cv2.imshow("Validation Feed (Chest ROI)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("\nShutting down validation system...")
            self.running = False
            self.cam.stop()
            cv2.destroyAllWindows()
            if self.ser: self.ser.close()

if __name__ == "__main__":
    print("--- Respiration Ground Truth Validation ---")
    
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
    
    plotter = ValidationRespPlotter(SERIAL_PORT, BAUD_RATE, cam)
    plotter.run()