from DataHandler import CameraHandler, WebcamHandler, IMUHandler
from Processor import FaceProcessor

XIAO_IP = "http://192.168.4.1"

ALGO_MAP = {
    "1": "GREEN",
    "2": "OMIT",
    "3": "POS_WANG",
    "4": "LMS",
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
    print("  1 - GREEN    (canal verde; Verkruysse 2008)")
    print("  2 - OMIT     (QR decomposition; Casado 2023)")
    print("  3 - POS_WANG (sliding window; Wang 2017)")
    print("  4 - LMS      (adaptive; requer IMU/drone)")
    algo_choice = input("Option [1-4]: ").strip()
    algo = ALGO_MAP.get(algo_choice, "GREEN")

    if algo == "LMS" and imu is None:
        print("Aviso: LMS requer IMU do drone. A usar GREEN.")
        algo = "GREEN"

    processor = FaceProcessor(cam, imu, algo=algo)
    processor.run()
