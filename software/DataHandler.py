import cv2
import threading
import time
import requests
import numpy as np

class CameraHandler:
    def __init__(self, base_url):
        """
        Initialises the camera handler and starts the capture thread.

        Parameters
        ----------
        base_url : str
            Base URL of the XIAO ESP32 camera (e.g. 'http://192.168.4.1').
        """
        self.base_url = base_url.rstrip("/")
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """
        Continuous capture thread with automatic reconnection.

        Reads the MJPEG stream, extracts individual JPEG frames from the
        binary buffer using the 0xFF 0xD8 (SOI) and 0xFF 0xD9 (EOI) markers,
        decodes them with OpenCV and stores the latest frame.
        """
        host = self.base_url.split("://", 1)[-1].split(":")[0]
        stream_url = f"http://{host}:81/stream"
        while self.running:
            try:
                response = requests.get(stream_url, stream=True, timeout=10)
                print(f"Stream ligado — Content-Type: {response.headers.get('Content-Type')}")
                buffer = bytes()
                for chunk in response.iter_content(chunk_size=4096):
                    if not self.running:
                        return
                    buffer += chunk
                    start = buffer.find(b'\xff\xd8')  # JPEG start marker
                    end   = buffer.find(b'\xff\xd9')  # JPEG end marker
                    if start == -1:
                        buffer = bytes()              # sem início — descarta tudo
                    elif end == -1:
                        buffer = buffer[start:]       # aguarda mais dados para completar o JPEG
                    elif end > start:
                        jpg = buffer[start:end+2]
                        buffer = buffer[end+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            if self.frame is None:
                                print("Primeiro frame recebido!")
                            self.frame = frame
            except Exception as e:
                print(f"Stream erro: {e} — a reconectar em 2s...")
                time.sleep(2)

    def get_frame(self):
        """
        Returns the latest captured frame.

        Returns
        -------
        np.ndarray or None
            Latest BGR frame, or None if no frame has been received yet.
        """
        return self.frame

    def stop(self):
        """Signals the capture thread to stop."""
        self.running = False


class WebcamHandler:
    def __init__(self, index=0):
        """
        Initialises the webcam handler and starts the capture thread.

        Parameters
        ----------
        index : int
            Webcam device index (0 = default PC webcam).
        """
        self.cap = cv2.VideoCapture(index)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Continuous capture thread. Reads frames from the webcam."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame is None:
                    print("First frame received!")
                self.frame = frame

    def get_frame(self):
        """
        Returns the latest captured frame.

        Returns
        -------
        np.ndarray or None
            Latest BGR frame, or None if no frame has been received yet.
        """
        return self.frame

    def stop(self):
        """Signals the capture thread to stop and releases the webcam."""
        self.running = False
        self.cap.release()


class IMUHandler:
    def __init__(self, base_url, poll_hz=10):
        """
        Initialises the IMU handler and starts the polling thread.

        Parameters
        ----------
        base_url : str
            Base URL of the XIAO ESP32 camera (e.g. 'http://192.168.4.1').
        poll_hz : float
            Polling frequency in Hz (default 10 Hz).
        """
        self.base_url = base_url.rstrip("/")
        self.imu = None   # latest dict: ax, ay, az, gx, gy, gz, bat
        self.running = True
        self._interval = 1.0 / poll_hz
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Polling thread. GETs /imu at the configured frequency and stores the result."""
        imu_url = f"{self.base_url}/imu"
        while self.running:
            try:
                response = requests.get(imu_url, timeout=2)
                data = response.json()
                if self.imu is None:
                    print("First IMU sample received!")
                self.imu = data
            except Exception as e:
                print(f"IMU error: {e}")
            time.sleep(self._interval)

    def get_imu(self):
        """
        Returns the latest IMU sample.

        Returns
        -------
        dict or None
            Latest sample with keys ax, ay, az (g), gx, gy, gz (deg/s), bat (mV),
            or None if no sample has been received yet.
        """
        return self.imu

    def stop(self):
        """Signals the polling thread to stop."""
        self.running = False
