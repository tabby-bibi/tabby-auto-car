import cv2
import numpy as np
import time
from picamera2 import Picamera2
from car_controller import CarController

# 관심영역(ROI) 설정 함수: 이미지 하단 40% 영역만 사용
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

# 히스토그램 계산 함수: 각 열의 픽셀 값을 합산하여 차선 중심을 찾음
def compute_histogram(img):
    return np.sum(img, axis=0)

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

            # 전처리
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 120)  # ✅ Canny 감도 조정

            roi = region_of_interest(edges)

            # 좌우 영역 나누기
            left_roi = roi[:, :img_center]
            right_roi = roi[:, img_center:]

            # 각 영역의 엣지 수 계산
            left_count = cv2.countNonZero(left_roi)
            right_count = cv2.countNonZero(right_roi)

            # 좌우 히스토그램 각각 계산
            left_hist = compute_histogram(left_roi)
            right_hist = compute_histogram(right_roi)

            if np.max(left_hist) > 0 and np.max(right_hist) > 0:
                # ✅ 좌우 중심이 모두 있을 경우 평균으로 중심 계산
                left_center = np.argmax(left_hist)
                right_center = np.argmax(right_hist) + img_center
                lane_center = (left_center + right_center) // 2
            else:
                # 한쪽 차선만 인식되면 전체 히스토그램 사용
                full_hist = compute_histogram(roi)
                lane_center = np.argmax(full_hist)

            offset = lane_center - img_center
            threshold = 30

            # ========================
            # 회전 방향 판단 로직
            # ========================

            if left_count == 0 and right_count == 0:
                direction = "stop"
                car.update(direction=None)
                car.stop_drive()

            elif abs(offset) < threshold:
                if abs(left_count - right_count) > 1000:
                    direction = "left" if left_count > right_count else "right"
                else:
                    direction = "straight"
                car.update(direction if direction != "straight" else None)
                car.set_motor_forward()

            else:
                direction = "left" if offset < -threshold else "right"
                car.update(direction)
                car.set_motor_forward()

            # ========================
            # 시각화
            # ========================

            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.line(frame, (lane_center, height), (lane_center, int(height * 0.6)), (255, 0, 0), 2)
            cv2.line(frame, (img_center, height), (img_center, int(height * 0.6)), (0, 255, 0), 2)

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
