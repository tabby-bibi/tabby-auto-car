from detect_object import ObjectDetector
import time

detector = ObjectDetector()

while True:
    status = detector.check_if_object_detected()
    if status == "clear":
        print("Straight")
    elif status == "stop":
        print("Stop")
    elif status == "warning":
        print("Warning")