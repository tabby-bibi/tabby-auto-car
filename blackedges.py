import cv2
import numpy as np
from picamera2 import Picamera2
from car_controller import CarController

# 관심 영역 설정 (하단 50%)
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

# 기울기 평균으로 방향 판단
def get_lane_direction_from_lines(lines):
    if lines is None:
        return "stop"

    slopes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # 수직선 제외
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

    if not slopes:
        return "stop"

    avg_slope = np.mean(slopes)
    print(f"Average slope: {avg_slope:.2f}")

    if abs(avg_slope) < 0.1:
        return "straight"
    elif avg_slope < 0:
        return "left"
    else:
        return "right"

def main():
    # 카메라 및 차량 초기화
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    car = CarController()

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]

            # 전처리
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 120)

            # ROI 설정
            roi = region_of_interest(edges)

            # HoughLinesP로 직선 검출
            lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50,
                                    minLineLength=50, maxLineGap=30)

            # 방향 판단
            direction = get_lane_direction_from_lines(lines)

            # 차량 제어
            if direction == "stop":
                car.update(direction=None)
                car.stop_drive()
            else:
                car.update(direction if direction != "straight" else None)
                car.set_motor_forward()

            # 시각화
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
