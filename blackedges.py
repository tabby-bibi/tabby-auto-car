import cv2
import numpy as np
from picamera2 import Picamera2
from car_controller import CarController

# ROI 설정 (하단 50%)
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (0, int(height * 0.5)),
        (width, int(height * 0.5)),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# 선의 기울기를 계산
def calculate_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf')  # 수직선
    return (y2 - y1) / (x2 - x1)

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (800, 600)})
    picam2.configure(config)
    picam2.start()

    car = CarController()

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            roi = region_of_interest(edges)

            lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=30)

            direction = "stop"
            left_slopes, right_slopes = [], []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = calculate_slope(x1, y1, x2, y2)
                    if slope == float('inf') or abs(slope) < 0.2:
                        continue  # 거의 수평 또는 수직선은 무시

                    if slope < 0:
                        left_slopes.append(slope)
                        color = (255, 0, 0)
                    else:
                        right_slopes.append(slope)
                        color = (0, 255, 0)

                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

                if left_slopes and right_slopes:
                    avg_left = np.mean(left_slopes)
                    avg_right = np.mean(right_slopes)
                    # 양쪽 모두 기울기가 크지 않으면 직진
                    if abs(avg_left) < 0.3 and abs(avg_right) < 0.3:
                        direction = "straight"
                    elif abs(avg_left) > abs(avg_right):
                        direction = "left"
                    else:
                        direction = "right"

                elif left_slopes:
                    direction = "left"
                elif right_slopes:
                    direction = "right"
                else:
                    direction = "stop"
            else:
                direction = "stop"

            # 차량 제어
            if direction == "straight":
                car.update(None)
                car.set_motor_forward()
            elif direction == "left":
                car.update("left")
                car.set_motor_forward()
            elif direction == "right":
                car.update("right")
                car.set_motor_forward()
            else:
                car.update(None)
                car.stop_drive()

            # 시각화
            cv2.putText(frame, f"Direction: {direction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.imshow("Lane Tracking", frame)
            cv2.imshow("ROI Edge", roi)

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
