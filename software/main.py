from DataHandler import CameraHandler, WebcamHandler, IMUHandler
from Processor import FaceProcessor
from ROIExtraction import ROI_FOREHEAD, ROI_FACE, ROI_MULTI

XIAO_IP = "http://192.168.4.1"

ALGO_MAP = {
    "1": "GREEN",
    "2": "OMIT",
    "3": "POS_WANG",
    "4": "LMS",
}

ROI_MAP = {
    "1": ROI_FOREHEAD,
    "2": ROI_FACE,
    "3": ROI_MULTI,
}

if __name__ == "__main__":
    print("Select video source:")
    print("  A - Drone camera (XIAO ESP32)")
    print("  B - PC webcam")
    choice = input("Option [A/B]: ").strip().upper()

    if choice == "B":
        cam = WebcamHandler(index=0)
        imu = None
    else:
        cam = CameraHandler(XIAO_IP)
        imu = IMUHandler(XIAO_IP)

    print("\nSelect algorithm:")
    print("  1 - GREEN    (green channel; Verkruysse 2008)")
    print("  2 - OMIT     (QR decomposition; Casado 2023)")
    print("  3 - POS_WANG (sliding window; Wang 2017)")
    print("  4 - LMS      (adaptive; requires IMU/drone)")
    algo_choice = input("Option [1-4]: ").strip()
    algo = ALGO_MAP.get(algo_choice, "GREEN")

    if algo == "LMS" and imu is None:
        print("Warning: LMS requires drone IMU. Falling back to GREEN.")
        algo = "GREEN"

    print("\nSelect ROI:")
    print("  1 - Forehead       (4 landmarks — original)")
    print("  2 - Full face      (36 landmarks — face oval)")
    print("  3 - Multi-region   (face oval + 9x9 grid + DMRS; Face2PPG)")
    roi_choice = input("Option [1-3]: ").strip()
    roi = ROI_MAP.get(roi_choice, ROI_FACE)

    processor = FaceProcessor(cam, imu, algo=algo, roi=roi)
    processor.run()
