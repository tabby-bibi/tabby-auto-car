import cv2
from picamera2 import Picamera2
import time

# 카메라 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

print("ESC를 누르면 종료됩니다.")

try:
    while True:
        frame = picam2.capture_array()
        cv2.imshow("Camera Preview", frame)

        # ESC 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
