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
        (0, int(height * 0.4)),  # 📌[개선] ROI 범위를 조금 더 위로 넓힘
        (width, int(height * 0.4)),
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
        while True:
            print("\n[Frame captured]")
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)  # 📌[개선] Canny threshold 조정

            roi = region_of_interest(edges)

            left_roi = roi[:, :img_center]
            right_roi = roi[:, img_center:]

            left_count = cv2.countNonZero(left_roi)
            right_count = cv2.countNonZero(right_roi)

            left_hist = compute_histogram(left_roi)
            right_hist = compute_histogram(right_roi)

            # 📌[개선] 더 신뢰도 있는 방식으로 차선 중심 계산
            left_max = np.max(left_hist)
            right_max = np.max(right_hist)

            center_found = False
            if left_max > 2000 and right_max > 2000:
                left_indices = np.where(left_hist > left_max * 0.6)[0]
                right_indices = np.where(right_hist > right_max * 0.6)[0]

                if len(left_indices) > 0 and len(right_indices) > 0:
                    left_center = int(np.mean(left_indices))
                    right_center = int(np.mean(right_indices)) + img_center
                    lane_center = (left_center + right_center) // 2
                    center_found = True
                    print(f"Left center: {left_center}, Right center: {right_center}, Lane center: {lane_center}")
                else:
                    print("One of the sides has no strong peak.")
            else:
                print("Insufficient edge strength on one or both sides.")

            offset = lane_center - img_center if center_found else 0
            print(f"Image center: {img_center}, Offset: {offset}")

            threshold = 30
            if not center_found:
                print("No reliable lane detected. Continuing straight or stopping.")
                direction = "stop"
                car.update(direction=None)
                car.stop_drive()
                continue  # 다음 프레임으로 넘어감

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

            # 검출된 차선 중앙 시각화 -> debugging용 (최종 결과는 시각화 안함)
            if center_found:
                cv2.circle(frame, (left_center, height - 10), 5, (255, 255, 0), -1)
                cv2.circle(frame, (right_center, height - 10), 5, (255, 255, 0), -1)
                cv2.line(frame, (lane_center, height), (lane_center, int(height * 0.6)), (255, 0, 0), 2)

            cv2.line(frame, (img_center, height), (img_center, int(height * 0.6)), (0, 255, 0), 2)
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Lane Tracking", frame)
            cv2.imshow("ROI Edge", roi)

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
