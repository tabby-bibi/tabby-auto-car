import cv2
import numpy as np
import time
from picamera2 import Picamera2
from car_controller import CarController

def filter_white(img):
    """HSV 색공간에서 흰색 검출"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 흰색 범위 설정 (조명에 따라 조정 가능)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    return cv2.bitwise_and(img, img, mask=mask)

def region_of_interest(img):
    height, width = img.shape[:2]
    roi = img[int(height * 2/3):, :]  # 하단 1/3만 사용
    mask = np.zeros_like(img)
    mask[int(height * 2/3):, :] = roi
    return mask

def decide_direction(center_lane_x, img_center, threshold=30):
    diff = center_lane_x - img_center
    if abs(diff) < threshold:
        return "straight"
    elif diff > 0:
        return "right"
    else:
        return "left"

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    car = CarController()

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            filtered = filter_white(frame)
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 10, 60)
            masked = region_of_interest(edges)

            lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=40)

            if lines is not None and len(lines) >= 2:
                # 가장 아래쪽 두 개 선을 찾음
                sorted_lines = sorted(lines, key=lambda l: min(l[0][1], l[0][3]), reverse=True)
                selected_lines = sorted_lines[:2]

                x_coords = []
                for line in selected_lines:
                    x1, y1, x2, y2 = line[0]
                    x_coords.append(x1)
                    x_coords.append(x2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)

                center_lane_x = sum(x_coords) // len(x_coords)
                direction = decide_direction(center_lane_x, img_center)

                if direction == "straight":
                    car.update(direction=None)
                    car.set_motor_forward()
                elif direction == "right":
                    car.update(direction="right")
                elif direction == "left":
                    car.update(direction="left")

                cv2.putText(frame, f"Direction: {direction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                car.stop_drive()
                cv2.putText(frame, "No lane detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Lane Detection", frame)
            cv2.imshow("Canny", edges)
            cv2.imshow("ROI", masked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
