"""
Respiratory rate estimation via the Bartula (2013) algorithm.

Bartula, M., Tigges, T., & Muehlsteff, J. "Camera-based System for
Contactless Monitoring of Respiration." IEEE EMBS, 2013.

Pipeline:
  Frame ROI → 1D vertical profile (mean + std per row, high-pass)
            → cross-correlation with previous profile (Hann window, sub-pixel)
            → shift integrated → position signal
            → global motion detection (block-based)
            → detrend + Butterworth bandpass [0.1–0.5 Hz]
            → peak detection + breath-by-breath validation
            → RR in rpm (EMA smoothed)

No pose estimation or body landmarks required. The camera only needs to
see any region of the chest/abdomen where respiratory motion is present.
"""

import time
import threading
import cv2
import mediapipe as mp
import numpy as np
from scipy import signal as sp_signal

_mp_pose    = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils

# ── Pose-based ROI: padding and height estimation from shoulder width ─────────
ROI_PAD_X_FACTOR  = 0.15   # horizontal padding as fraction of shoulder width
ROI_HEIGHT_FACTOR = 1.20   # estimated chest height as fraction of shoulder width
ROI_TOP_FACTOR    = 0.10   # how far above shoulder line the ROI starts

# ── Motion detection ──────────────────────────────────────────────────────────
BLOCK_SIZE    = 16    # pixels per block (for global motion detector)
MOTION_THRESH = 12    # mean absolute diff per block to classify as "moving"
GLOBAL_MOTION_RATIO = 0.30  # fraction of moving blocks that flags global motion

# ── Profile high-pass ─────────────────────────────────────────────────────────
HP_KERNEL = 21        # moving-average kernel size for profile high-pass filter

# ── RR estimation ─────────────────────────────────────────────────────────────
MIN_SEC       = 20    # minimum seconds of signal before first estimate
RR_ALPHA      = 0.15  # EMA smoothing for display
MIN_BREATH_S  = 1.5   # shortest valid breath (s)   → ~40 rpm max
MAX_BREATH_S  = 10.0  # longest  valid breath (s)   →  6 rpm min


class RespiratoryProcessor:
    def __init__(self, camera, display: bool = True):
        """
        Parameters
        ----------
        camera : CameraHandler or WebcamHandler
            Video source.
        display : bool
            Show camera window and matplotlib plots (default True).
        """
        self.camera  = camera
        self.display = display

        self.position_signal  = []   # integrated chest position per frame
        self.frame_timestamps = []
        self.motion_flags     = []   # True when global motion detected on that frame

        self._prev_profile    = None
        self._prev_gray_roi   = None
        self._position        = 0.0  # running integral of frame-to-frame shifts
        self._fixed_roi       = None  # (y1, y2, x1, x2) set by Pose at startup

        self.rr_estimate  = None
        self.rr_ema       = None
        self._rr_computing = False
        self._rr_lock      = threading.Lock()

    # ── Timing ────────────────────────────────────────────────────────────────

    def get_fps(self, window=60):
        """Estimates FPS from recent timestamps; fallback 25.0 Hz."""
        if len(self.frame_timestamps) < 2:
            return 25.0
        recent = self.frame_timestamps[-window:]
        return (len(recent) - 1) / (recent[-1] - recent[0])

    # ── ROI helpers ───────────────────────────────────────────────────────────

    def _roi_bounds(self, frame):
        """Returns (y1, y2, x1, x2) pixel coordinates of the Pose-derived chest ROI."""
        return self._fixed_roi

    def _init_roi_from_pose(self, timeout=15):
        """
        Runs MediaPipe Pose on live frames until both shoulders are detected
        with visibility > 0.5, then computes a tight chest ROI from the
        shoulder landmarks. Closes Pose immediately afterwards.

        ROI boundaries:
        - Top    : shoulder_y − ROI_TOP_FACTOR × shoulder_width
        - Bottom : hip_y if hips visible, else shoulder_y + ROI_HEIGHT_FACTOR × shoulder_width
        - Left/Right : shoulder edges ± ROI_PAD_X_FACTOR × shoulder_width

        Falls back to the default frame-fraction ROI if detection times out.

        Parameters
        ----------
        timeout : float   Maximum seconds to wait for a valid detection.
        """
        print("Detecting thorax position with Pose... (point the camera at the chest)")

        pose = _mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        L_SH = _mp_pose.PoseLandmark.LEFT_SHOULDER.value
        R_SH = _mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        L_HP = _mp_pose.PoseLandmark.LEFT_HIP.value
        R_HP = _mp_pose.PoseLandmark.RIGHT_HIP.value

        t_start = time.time()
        while time.time() - t_start < timeout:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res   = pose.process(rgb)
            rgb.flags.writeable = True

            if self.display:
                preview = frame.copy()
                cv2.putText(preview, "Detecting thorax... please wait",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                if self.display:
                    _mp_drawing.draw_landmarks(
                        preview, res.pose_landmarks, _mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=_mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=3),
                        connection_drawing_spec=_mp_drawing.DrawingSpec(
                            color=(0, 200, 255), thickness=1),
                    )

                if (lm[L_SH].visibility > 0.5 and lm[R_SH].visibility > 0.5):
                    sx1 = int(min(lm[L_SH].x, lm[R_SH].x) * w)
                    sx2 = int(max(lm[L_SH].x, lm[R_SH].x) * w)
                    sy  = int((lm[L_SH].y + lm[R_SH].y) / 2 * h)
                    sw  = max(sx2 - sx1, 1)                    # shoulder width

                    pad_x = int(sw * ROI_PAD_X_FACTOR)
                    x1    = max(0, sx1 - pad_x)
                    x2    = min(w, sx2 + pad_x)
                    y1    = max(0, sy - int(sw * ROI_TOP_FACTOR))

                    if lm[L_HP].visibility > 0.5 and lm[R_HP].visibility > 0.5:
                        hip_y = int((lm[L_HP].y + lm[R_HP].y) / 2 * h)
                        y2    = min(h, hip_y)
                    else:
                        y2    = min(h, sy + int(sw * ROI_HEIGHT_FACTOR))

                    if self.display:
                        # Draw detected ROI and hold for confirmation
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(preview, "ROI detected — starting",
                                    (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow('Respiratory Rate — Bartula 2013', preview)
                        cv2.waitKey(1000)                      # show ROI for 1 s

                    self._fixed_roi = (y1, y2, x1, x2)
                    print(f" ROI: y=[{y1},{y2}] x=[{x1},{x2}]")
                    pose.close()
                    return

            if self.display:
                cv2.imshow('Respiratory Rate — Bartula 2013', preview)
                cv2.waitKey(1)

        pose.close()
        raise RuntimeError(
            "Timeout: no person detected in {:.0f} s. "
            "Make sure the camera is pointing at the chest.".format(timeout)
        )

    # ── Profile extraction ────────────────────────────────────────────────────

    @staticmethod
    def _make_profile(roi_gray):
        """
        Collapses a grayscale ROI onto a 1D vertical axis.

        profile[i] = mean(row_i) + std(row_i)

        A high-pass filter (subtract moving average) is applied to make the
        profile insensitive to global illumination changes and to enhance
        edges — the features that shift most clearly with chest movement.

        Parameters
        ----------
        roi_gray : np.ndarray, shape (H, W)   Grayscale ROI.

        Returns
        -------
        np.ndarray, shape (H,)   High-passed vertical profile.
        """
        profile = roi_gray.mean(axis=1) + roi_gray.std(axis=1)
        kernel  = np.ones(HP_KERNEL) / HP_KERNEL
        smooth  = np.convolve(profile, kernel, mode='same')
        return profile - smooth

    # ── Cross-correlation ─────────────────────────────────────────────────────

    @staticmethod
    def _profile_shift(p_curr, p_prev):
        """
        Estimates the vertical translatory shift (in pixels) between two
        profiles via phase cross-correlation.

        Uses a Hann window before the FFT to reduce boundary/spectral leakage
        effects. Sub-pixel accuracy is obtained by quadratic interpolation
        around the correlation peak.

        Cross-correlation:
            r = F⁻¹( F(p_curr) · conj(F(p_prev)) )

        Parameters
        ----------
        p_curr : np.ndarray   Current frame profile.
        p_prev : np.ndarray   Previous frame profile.

        Returns
        -------
        float   Signed shift in pixels (positive = profile moved downward).
        """
        n     = len(p_curr)
        win   = np.hanning(n)
        P_c   = np.fft.rfft(p_curr * win)
        P_p   = np.fft.rfft(p_prev * win)
        r     = np.fft.irfft(P_c * np.conj(P_p), n=n)
        r     = np.fft.fftshift(r)          # centre at zero lag
        peak  = int(np.argmax(r))

        # Quadratic sub-pixel interpolation
        if 0 < peak < len(r) - 1:
            y0, y1, y2 = r[peak - 1], r[peak], r[peak + 1]
            denom = 2.0 * (y0 - 2 * y1 + y2)
            delta = (y0 - y2) / (denom + 1e-9)
        else:
            delta = 0.0

        return (peak + delta) - n // 2

    # ── Global motion detection ───────────────────────────────────────────────

    @staticmethod
    def _global_motion(curr_gray, prev_gray):
        """
        Divides the ROI into BLOCK_SIZE×BLOCK_SIZE blocks and counts those
        whose mean absolute frame difference exceeds MOTION_THRESH.

        Returns True when the fraction of moving blocks exceeds
        GLOBAL_MOTION_RATIO, indicating a large non-respiratory movement
        (e.g. the subject repositioning or an arm crossing the ROI).

        Parameters
        ----------
        curr_gray : np.ndarray  Current ROI (grayscale).
        prev_gray : np.ndarray  Previous ROI (grayscale).

        Returns
        -------
        bool
        """
        diff      = np.abs(curr_gray.astype(np.float32) -
                           prev_gray.astype(np.float32))
        h, w      = diff.shape
        n_moving  = 0
        n_total   = 0
        for y in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
            for x in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
                n_total += 1
                if diff[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE].mean() > MOTION_THRESH:
                    n_moving += 1
        return (n_moving / n_total) > GLOBAL_MOTION_RATIO if n_total else False

    # ── Frame processing ──────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Processes one frame: extracts the vertical profile from the ROI,
        computes the cross-correlation shift relative to the previous frame,
        integrates the shift into the chest position signal, and runs the
        global motion detector.

        A sample is appended on every frame after the first (no landmark
        visibility requirement — works as long as the ROI covers the chest).

        Parameters
        ----------
        frame : np.ndarray  BGR frame from the camera.

        Returns
        -------
        np.ndarray  Frame annotated with the ROI rectangle and motion flag.
        """
        gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y1, y2, x1, x2 = self._roi_bounds(frame)
        roi_gray       = gray[y1:y2, x1:x2]
        profile        = self._make_profile(roi_gray)

        if self._prev_profile is not None and self._prev_gray_roi is not None:
            shift          = self._profile_shift(profile, self._prev_profile)
            self._position += shift
            motion         = self._global_motion(roi_gray, self._prev_gray_roi)

            self.position_signal.append(self._position)
            self.frame_timestamps.append(time.time())
            self.motion_flags.append(motion)

        self._prev_profile  = profile
        self._prev_gray_roi = roi_gray

        # Draw ROI overlay
        motion_now = self.motion_flags[-1] if self.motion_flags else False
        color      = (0, 80, 255) if motion_now else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "ROI [MOTION]" if motion_now else "ROI"
        cv2.putText(frame, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    # ── Signal processing ─────────────────────────────────────────────────────

    @staticmethod
    def apply_filters(sig, fs):
        """
        Linear detrend + Butterworth bandpass [0.1–0.5 Hz].

        Detrend removes accumulated drift from the position integral.
        Bandpass isolates the respiratory frequency band (6–30 rpm).
        """
        sig  = sp_signal.detrend(sig.astype(np.double))
        nyq  = fs / 2
        lo   = max(0.1 / nyq, 0.001)
        hi   = min(0.5 / nyq, 0.999)
        b, a = sp_signal.butter(2, [lo, hi], btype='bandpass')
        return sp_signal.filtfilt(b, a, sig)

    @staticmethod
    def _estimate_rr_from_peaks(filtered, fs, motion_flags):
        """
        Detects inhalation peaks in the filtered position signal and validates
        each breath candidate individually (Bartula breath-by-breath approach).

        Validation criteria per inter-peak interval:
        - Duration in [MIN_BREATH_S, MAX_BREATH_S]
        - Fraction of frames flagged as global motion < 0.5

        RR is the median of valid inter-peak intervals converted to rpm.
        Returns None if fewer than 2 valid intervals are found or the result
        falls outside [6, 60] rpm.

        Parameters
        ----------
        filtered     : np.ndarray  Bandpass-filtered position signal.
        fs           : float       Sampling frequency (Hz).
        motion_flags : list[bool]  Per-frame global motion flags (same length).

        Returns
        -------
        float or None   RR in breaths per minute.
        """
        min_dist = int(MIN_BREATH_S * fs)
        peaks, _ = sp_signal.find_peaks(
            filtered,
            distance=min_dist,
            prominence=np.std(filtered) * 0.25,
        )

        if len(peaks) < 2:
            return None

        good_intervals = []
        for i in range(len(peaks) - 1):
            p1, p2       = peaks[i], peaks[i + 1]
            interval_sec = (p2 - p1) / fs
            seg_motion   = motion_flags[p1:p2]
            motion_ratio = sum(seg_motion) / len(seg_motion) if seg_motion else 0.0

            if MIN_BREATH_S <= interval_sec <= MAX_BREATH_S and motion_ratio < 0.5:
                good_intervals.append(interval_sec)

        if len(good_intervals) < 2:
            return None

        rr = 60.0 / float(np.median(good_intervals))
        return rr if 6.0 <= rr <= 60.0 else None

    def _compute_rr_background(self):
        """Runs RR estimation in a background thread every 30 frames."""
        fs     = self.get_fps()
        window = max(int(MIN_SEC * fs), 64)
        with self._rr_lock:
            n = len(self.position_signal)
        if n < window:
            with self._rr_lock:
                self._rr_computing = False
            return

        sig   = np.array(self.position_signal[-window:])
        flags = list(self.motion_flags[-window:])
        try:
            filtered = self.apply_filters(sig, fs)
            rr       = self._estimate_rr_from_peaks(filtered, fs, flags)
        except Exception:
            with self._rr_lock:
                self._rr_computing = False
            return

        with self._rr_lock:
            if rr is not None:
                if self.rr_ema is None:
                    self.rr_ema = rr
                else:
                    self.rr_ema = RR_ALPHA * rr + (1 - RR_ALPHA) * self.rr_ema
                self.rr_estimate = self.rr_ema
            self._rr_computing = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Acquisition, display and real-time plot loop. Press 'q' to quit."""
        self._init_roi_from_pose()

        if self.display:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, (ax_raw, ax_filt) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            fig.suptitle('Bartula Respiratory Signal')
            line_raw,  = ax_raw.plot([], [],  color='#00bcd4', linewidth=0.8)
            line_filt, = ax_filt.plot([], [], color='#ff9100', linewidth=0.8)
            ax_raw.set_ylabel('Position (integrated shift, px)')
            ax_filt.set_ylabel('Filtered [0.1–0.5 Hz]')
            ax_filt.set_xlabel('Frame')
            fig.tight_layout()

        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                annotated = self.process_frame(frame)
                n         = len(self.position_signal)

                if n > 0 and not self._rr_computing and n % 30 == 0:
                    with self._rr_lock:
                        self._rr_computing = True
                    threading.Thread(target=self._compute_rr_background,
                                     daemon=True).start()

                # Terminal status
                motion_now = self.motion_flags[-1] if self.motion_flags else False
                motion_str = " [MOTION]" if motion_now else ""
                if self.rr_estimate is not None:
                    print(
                        f"\rFrames: {n} | fs: {self.get_fps():.1f} Hz"
                        f" | RR: {self.rr_estimate:.1f} rpm{motion_str}      ",
                        end="", flush=True,
                    )
                else:
                    needed = max(int(MIN_SEC * self.get_fps()), 64)
                    pct    = min(100, int(n / needed * 100))
                    print(f"\rFrames: {n} | Collecting... {pct}%{motion_str}   ",
                          end="", flush=True)

                if self.display:
                    # Update plots every 5 frames
                    if n % 5 == 0 and n > 0:
                        WIN = 500
                        raw = np.array(self.position_signal[-WIN:])
                        x   = np.arange(max(0, n - WIN), n)
                        line_raw.set_xdata(x)
                        line_raw.set_ydata(raw)
                        ax_raw.set_xlim(x[0], x[-1] + 1)
                        ax_raw.relim()
                        ax_raw.autoscale_view(scalex=False, scaley=True)

                        if len(raw) >= 64:
                            try:
                                filt = self.apply_filters(raw, self.get_fps())
                                line_filt.set_xdata(x)
                                line_filt.set_ydata(filt)
                                ax_filt.relim()
                                ax_filt.autoscale_view(scalex=False, scaley=True)
                            except Exception:
                                pass

                        fig.canvas.draw()
                        fig.canvas.flush_events()

                    cv2.imshow('Respiratory Rate — Bartula 2013', annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop()
            if self.display:
                plt.ioff()
                plt.show()

    # ── Teardown ──────────────────────────────────────────────────────────────

    def stop(self):
        """Stops the camera and closes all windows."""
        self.camera.stop()
        cv2.destroyAllWindows()
        print(f"\nPosition signal collected: {len(self.position_signal)} frames")
