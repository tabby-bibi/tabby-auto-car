import cv2
import numpy as np
import time
from picamera2 import Picamera2
from car_controller import CarController

# 관심영역(ROI) 설정 함수
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


# 히스토그램 계산 함수
def compute_histogram(img):
    return np.sum(img, axis=0)

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)},
        controls={"FrameRate": 15}
    )
    picam2.configure(config)

    picam2.set_controls({
        "AwbEnable": False,
        "AeEnable": False,
        "ExposureTime": 10000,
        "AnalogueGain": 1.0,
        "ColourGains": (1.5, 1.5)
    })

    picam2.start()
    car = CarController()

    try:
        lane_centers = []

        while True:
            print("\n[Frame captured]")
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            # HSV 변환 및 색상 기반 차선 추출
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))       # 흰색 범위
            yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))   # 노란색 범위
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

            print("Applied HSV masking")

            # Canny edge detection
            edges = cv2.Canny(combined_mask, 50, 150)
            print("Canny edges detected")

            # ROI 적용
            roi = region_of_interest(edges)
            print("ROI extracted")

            # 왼쪽/오른쪽 나누기
            left_roi = roi[:, :img_center]
            right_roi = roi[:, img_center:]

            left_count = cv2.countNonZero(left_roi)
            right_count = cv2.countNonZero(right_roi)
            print(f"Left edge count: {left_count}, Right edge count: {right_count}")

            left_hist = compute_histogram(left_roi)
            right_hist = compute_histogram(right_roi)

            if np.max(left_hist) > 0 and np.max(right_hist) > 0:
                left_center = np.argmax(left_hist)
                right_center = np.argmax(right_hist) + img_center
                lane_center = (left_center + right_center) // 2
                print(f"Left center: {left_center}, Right center: {right_center}, Lane center: {lane_center}")
            else:
                full_hist = compute_histogram(roi)
                lane_center = np.argmax(full_hist)
                print("Only one side detected. Lane center from full ROI:", lane_center)

            # 최근 프레임 평균 (더 안정화)
            lane_centers.append(lane_center)
            if len(lane_centers) > 3:
                lane_centers.pop(0)
            lane_center = int(np.mean(lane_centers))

            offset = lane_center - img_center
            print(f"Image center: {img_center}, Offset: {offset}")

            threshold = 30
            if left_count == 0 and right_count == 0:
                print("No lane detected. Stopping.")
                direction = "stop"
                car.update(direction=None)
                car.stop_drive()

            elif abs(offset) < threshold:
                if abs(left_count - right_count) > 4000:
                    direction = "right" if left_count > right_count else "left"
                    print(f"Offset small, but edge imbalance. Adjusting {direction}.")
                else:
                    direction = "straight"
                    print("Going straight.")

                if direction == "straight":
                    car.update(direction=None)
                    car.set_motor_forward()
                else:
                    car.update(direction)

            else:
                direction = "right" if offset < -threshold else "left"
                print(f"Offset large. Turning {direction}.")
                car.update(direction)

            cv2.imshow("Lane Tracking", frame)
            cv2.imshow("ROI Edge", roi)
            cv2.imshow("Combined Mask", combined_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")

    finally:
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
