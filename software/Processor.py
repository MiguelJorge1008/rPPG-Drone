import time
import threading
import cv2
import mediapipe as mp
import numpy as np
from scipy import signal as sp_signal
from scipy import sparse
from DataHandler import CameraHandler
from algoritmos.green import GREEN
from algoritmos.omit import OMIT
from algoritmos.pos_wang import POS_WANG
from algoritmos.adaptive_lms import ADAPTIVE_LMS
from ROIExtraction import (ROI_FOREHEAD, ROI_FACE, ROI_MULTI,
                            GRID_N, R_MAX, MIN_REGION_PIXELS,
                            get_roi_polygon, extract_grid_rgb,
                            dmrs_select, draw_grid_overlay)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

LANDMARK_ALPHA = 0.35
HR_ALPHA       = 0.1


class FaceProcessor:
    def __init__(self, camera: CameraHandler, imu=None, algo: str = "GREEN",
                 roi: str = ROI_FACE):
        """
        Initialises the FaceProcessor.

        Parameters
        ----------
        camera : CameraHandler or WebcamHandler
            Video source.
        imu : IMUHandler or None
            IMU source (drone only). None when using webcam.
        algo : str
            Algorithm to use: "GREEN", "OMIT", "POS_WANG", or "LMS".
        """
        self.camera = camera
        self.imu    = imu
        self.algo   = algo
        self.roi    = roi

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.rgb_signal          = []   # list of [R_mean, G_mean, B_mean] per frame (full face)
        self.region_rgb          = []   # list of (K, 3) arrays per frame — grid regions
        self.frame_timestamps    = []
        self.hr_estimate         = None
        self.landmark_ema        = None
        self._last_face_detected = False
        self.hr_ema           = None
        self.imu_signal       = []
        self._last_imu_id     = None
        self._hr_computing    = False   # flag: background HR thread running
        self._hr_lock         = threading.Lock()
        self._latest_bvps     = None
        self._bvps_fresh      = False   # True only when new BVP result not yet plotted
        self._selected_regions = []     # DMRS-selected region indices (for visualisation)

    # ── Face and ROI ──────────────────────────────────────────────────────────


    def process_frame(self, frame):
        """
        Runs FaceMesh on a single frame, extracts the forehead ROI mean RGB
        and annotates the frame.

        Timestamps are recorded only when a valid ROI is found, keeping
        frame_timestamps and rgb_signal in sync.

        Parameters
        ----------
        frame : np.ndarray  BGR frame from the camera.

        Returns
        -------
        np.ndarray  Annotated BGR frame.
        """
        small = cv2.resize(frame, (320, 240))
        h, w  = small.shape[:2]
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        self._last_face_detected = results.multi_face_landmarks is not None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=small,
                    landmark_list=face_landmarks,
                    connections=None,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=-1, circle_radius=1
                    )
                )
                poly, self.landmark_ema = get_roi_polygon(
                    face_landmarks, w, h, self.landmark_ema, LANDMARK_ALPHA,
                    roi_mode=self.roi
                )

                face_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(face_mask, [poly], 255)
                roi_pixels = rgb[face_mask == 255]
                if len(roi_pixels) > 0:
                    face_mean = roi_pixels.mean(axis=0)
                    self.rgb_signal.append(face_mean)
                    self.frame_timestamps.append(time.time())
                    if self.roi == ROI_MULTI:
                        self.region_rgb.append(
                            extract_grid_rgb(rgb, face_mask, poly, h, w, face_mean)
                        )

                cv2.polylines(small, [poly], isClosed=True, color=(0, 255, 255), thickness=1)
                if self.roi == ROI_MULTI:
                    with self._hr_lock:
                        sel = list(self._selected_regions)
                    draw_grid_overlay(small, poly, h, w, sel)
                else:
                    overlay = small.copy()
                    cv2.fillPoly(overlay, [poly], color=(0, 255, 255))
                    cv2.addWeighted(overlay, 0.2, small, 0.8, 0, small)

            frame = cv2.resize(small, (frame.shape[1], frame.shape[0]))
        return frame

    # ── Signal processing ─────────────────────────────────────────────────────

    def get_fps(self, window=60):
        """
        Estimates fps from the last `window` face-detected frame timestamps.
        Falls back to 27.0 Hz if fewer than 2 timestamps are available.
        """
        if len(self.frame_timestamps) < 2:
            return 27.0
        recent = self.frame_timestamps[-window:]
        return (len(recent) - 1) / (recent[-1] - recent[0])

    def _estimate_hr_realtime(self, min_sec=30):
        """
        Computes BVP signals and HR estimates for all available algorithms
        on a sliding window of the last min_sec seconds.

        Returns None if insufficient data has been collected.

        Returns
        -------
        tuple (hrs, bvps) or None
            hrs  — dict of BPM values per algorithm
            bvps — dict of filtered BVP arrays per algorithm
        """
        fs     = self.get_fps()
        window = max(int(min_sec * fs), 64)
        if len(self.rgb_signal) < window:
            return None

        # Multi-region: use last DMRS-selected regions (updated separately)
        if self.roi == ROI_MULTI:
            with self._hr_lock:
                sel = list(self._selected_regions)
            if sel and len(self.region_rgb) >= window:
                region_win = np.array(self.region_rgb[-window:])
                rgb_win    = region_win[:, sel, :].mean(axis=1)
            else:
                rgb_win = np.array(self.rgb_signal[-window:])
            selected = sel
        else:
            rgb_win  = np.array(self.rgb_signal[-window:])
            selected = []

        if self.algo == 'GREEN':
            raw = GREEN(rgb_win)
        elif self.algo == 'OMIT':
            raw = OMIT(rgb_win)
        elif self.algo == 'POS_WANG':
            raw = POS_WANG(rgb_win, fs)
        elif self.algo == 'LMS':
            if len(self.imu_signal) < 2:
                return None
            ratio   = len(self.imu_signal) / len(self.rgb_signal)
            imu_win = self.imu_signal[max(0, int(len(self.imu_signal) - window * ratio)):]
            raw = ADAPTIVE_LMS(rgb_win, imu_win)
        else:
            return None

        bvp  = self.apply_filters(raw, fs)
        bvps = {self.algo: bvp}
        hrs  = {self.algo: self.estimate_hr(bvp, fs)}
        return hrs, bvps, selected

    @staticmethod
    def estimate_hr(bvp, fs):
        """
        Estimates heart rate from a filtered BVP signal via FFT peak detection.

        Parameters
        ----------
        bvp : np.ndarray, shape (N,)  Filtered BVP signal.
        fs  : float  Sampling frequency in Hz.

        Returns
        -------
        float  Heart rate in BPM.
        """
        freqs = np.fft.rfftfreq(len(bvp), d=1 / fs)
        power = np.abs(np.fft.rfft(bvp)) ** 2
        band  = (freqs >= 0.75) & (freqs <= 4.0)
        return freqs[band][np.argmax(power[band])] * 60

    @staticmethod
    def apply_filters(bvp, fs):
        """
        Detrend + Butterworth bandpass [0.75-4.0 Hz].
        Pre/post-processing pipeline as per Face2PPG (Casado & Lopez, 2023).

        Parameters
        ----------
        bvp : np.ndarray, shape (N,)  Raw BVP signal.
        fs  : float  Sampling frequency in Hz.

        Returns
        -------
        np.ndarray, shape (N,)  Filtered BVP signal.
        """
        n     = len(bvp)
        H     = np.identity(n)
        diags = np.array([np.ones(n), -2 * np.ones(n), np.ones(n)])
        D     = sparse.spdiags(diags, [0, 1, 2], n - 2, n).toarray()
        bvp   = np.dot(H - np.linalg.inv(H + 100 ** 2 * D.T @ D), bvp)
        nyq  = fs / 2
        lo   = max(0.75 / nyq, 0.001)
        hi   = min(4.0  / nyq, 0.999)
        b, a  = sp_signal.butter(2, [lo, hi], btype='bandpass')
        return sp_signal.filtfilt(b, a, bvp.astype(np.double))

    def _compute_dmrs_background(self):
        """Runs DMRS region selection in a background thread (every 90 frames)."""
        fs     = self.get_fps()
        window = max(int(30 * fs), 64)
        if len(self.region_rgb) < window:
            with self._hr_lock:
                self._hr_computing = False
            return
        region_win = np.array(self.region_rgb[-window:])
        green_win  = region_win[:, :, 1]
        selected   = dmrs_select(green_win, fs, r_max=R_MAX)
        with self._hr_lock:
            self._selected_regions = selected
            self._hr_computing = False

    def _compute_hr_background(self):
        """Runs HR estimation in a background thread and stores the result."""
        result = self._estimate_hr_realtime()
        with self._hr_lock:
            if result is not None:
                raw, bvps, selected = result
                if self.hr_ema is None:
                    self.hr_ema = raw
                else:
                    self.hr_ema = {
                        k: HR_ALPHA * raw[k] + (1 - HR_ALPHA) * self.hr_ema[k]
                        for k in raw
                    }
                self.hr_estimate = self.hr_ema
                self._latest_bvps = bvps
                self._bvps_fresh  = True
            self._hr_computing = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Acquisition, display and real-time plot loop. Press 'q' to quit."""
        import matplotlib.pyplot as plt

        plt.ion()

        # RGB figure
        fig_rgb, axes_rgb = plt.subplots(3, 1, figsize=(8, 4), sharex=True)
        fig_rgb.suptitle('RGB Signal')
        lines_rgb = []
        for ax, color, label in zip(axes_rgb, ['#ff4444', '#44dd88', '#4488ff'], ['R', 'G', 'B']):
            line, = ax.plot([], [], color=color, linewidth=0.8)
            ax.set_ylabel(label)
            lines_rgb.append(line)
        axes_rgb[-1].set_xlabel('Frame')
        fig_rgb.tight_layout()

        # BVP figure (algorithm selected at startup)
        _algo_colors = {'GREEN': '#00e676', 'OMIT': '#448aff',
                        'POS_WANG': '#ff9100', 'LMS': '#e040fb'}
        fig_bvp, ax_bvp = plt.subplots(1, 1, figsize=(8, 2))
        fig_bvp.suptitle('BVP Signal')
        line_bvp, = ax_bvp.plot([], [], color=_algo_colors.get(self.algo, '#00e676'), linewidth=0.8)
        ax_bvp.set_ylabel(self.algo)
        ax_bvp.set_xlabel('Frame (window)')
        lines_bvp = [line_bvp]
        fig_bvp.tight_layout()

        # IMU figure
        if self.imu is not None:
            fig_imu, axes_imu = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
            fig_imu.suptitle('IMU — MPU-6050')
            lines_accel, lines_gyro = [], []
            for color, label in zip(['#ff4444', '#44dd88', '#4488ff'], ['ax', 'ay', 'az']):
                l, = axes_imu[0].plot([], [], color=color, linewidth=0.8, label=label)
                lines_accel.append(l)
            axes_imu[0].set_ylabel('Accel (g)')
            axes_imu[0].legend(fontsize=7)
            for color, label in zip(['#ff4444', '#44dd88', '#4488ff'], ['gx', 'gy', 'gz']):
                l, = axes_imu[1].plot([], [], color=color, linewidth=0.8, label=label)
                lines_gyro.append(l)
            axes_imu[1].set_ylabel('Gyro (°/s)')
            axes_imu[1].set_xlabel('Sample')
            axes_imu[1].legend(fontsize=7)
            fig_imu.tight_layout()

        HR_ALPHA = 0.1

        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                # Collect IMU sample (deduplicated by object identity)
                if self.imu is not None:
                    sample = self.imu.get_imu()
                    if sample is not None and id(sample) != self._last_imu_id:
                        self.imu_signal.append(sample)
                        self._last_imu_id = id(sample)

                annotated = self.process_frame(frame)

                n_frames = len(self.rgb_signal)
                if n_frames > 0 and not self._hr_computing:
                    if self.roi == ROI_MULTI and n_frames % 90 == 0:
                        # DMRS every 90 frames — updates selected regions
                        with self._hr_lock:
                            self._hr_computing = True
                        threading.Thread(target=self._compute_dmrs_background,
                                         daemon=True).start()
                    elif n_frames % 30 == 0:
                        # HR every 30 frames
                        with self._hr_lock:
                            self._hr_computing = True
                        threading.Thread(target=self._compute_hr_background,
                                         daemon=True).start()

                # Update BVP plots only when background thread produced new results
                with self._hr_lock:
                    if self._bvps_fresh:
                        bvps = self._latest_bvps
                        self._bvps_fresh = False
                    else:
                        bvps = None
                if bvps is not None:
                    for line, (name, bvp) in zip(lines_bvp, bvps.items()):
                        line.set_xdata(np.arange(len(bvp)))
                        line.set_ydata(bvp)
                        ax = line.axes
                        ax.relim()
                        ax.autoscale_view()
                    fig_bvp.canvas.draw()
                    fig_bvp.canvas.flush_events()

                # HR print to terminal (camera display disabled)
                face_str = "SIM" if self._last_face_detected else "NÃO"
                roi_str  = f"ROI: {len(self.rgb_signal)} frames | fs: {self.get_fps():.1f} Hz"
                if self.hr_estimate is not None:
                    for name, hr in self.hr_estimate.items():
                        print(f"\rFace: {face_str} | {roi_str} | {name}: {hr:.1f} BPM      ", end="", flush=True)
                else:
                    n      = len(self.rgb_signal)
                    needed = max(int(30 * self.get_fps()), 64)
                    pct    = min(100, int(n / needed * 100))
                    print(f"\rFace: {face_str} | {roi_str} | A recolher... {pct}%   ", end="", flush=True)

                cv2.imshow('rPPG', annotated)

                # Update RGB + IMU every 5 frames
                if len(self.rgb_signal) % 5 == 0 and len(self.rgb_signal) > 0:
                    sig = np.array(self.rgb_signal)
                    n   = len(sig)
                    x   = np.arange(n)
                    for i, line in enumerate(lines_rgb):
                        line.set_xdata(x)
                        line.set_ydata(sig[:, i])
                    for ax in axes_rgb:
                        ax.set_xlim(max(0, n - 300), max(300, n))
                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=True)
                    fig_rgb.canvas.draw()
                    fig_rgb.canvas.flush_events()

                    if self.imu is not None and self.imu_signal:
                        WIN  = 300
                        data = self.imu_signal[-WIN:]
                        n_i  = len(data)
                        x_i  = np.arange(n_i)
                        for line, field in zip(lines_accel, ['ax', 'ay', 'az']):
                            line.set_xdata(x_i)
                            line.set_ydata([s[field] for s in data])
                        for line, field in zip(lines_gyro, ['gx', 'gy', 'gz']):
                            line.set_xdata(x_i)
                            line.set_ydata([s[field] for s in data])
                        for ax in axes_imu:
                            ax.relim()
                            ax.autoscale_view()
                        fig_imu.canvas.draw()
                        fig_imu.canvas.flush_events()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop()
            plt.ioff()
            plt.show()

    # ── Teardown ──────────────────────────────────────────────────────────────

    def stop(self):
        """Closes FaceMesh, the camera and the IMU."""
        self.face_mesh.close()
        self.camera.stop()
        if self.imu is not None:
            self.imu.stop()
        cv2.destroyAllWindows()
        if self.rgb_signal:
            print(f"RGB signal collected: {len(self.rgb_signal)} frames")
