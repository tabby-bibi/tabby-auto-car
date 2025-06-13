import cv2
import numpy as np
import time
from picamera2 import Picamera2
from car_controller import CarController
import matplotlib.pyplot as plt


# 관심영역(ROI) 설정 함수: 이미지 하단 50% 영역만 사용
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


# 히스토그램 시각화 함수 (필요시 사용)
def show_histogram(hist, title="Histogram"):
    plt.figure(figsize=(6, 3))
    plt.title(title)
    plt.xlabel("Pixel Position (X)")
    plt.ylabel("Edge Intensity")
    plt.plot(hist, color='blue')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)

    # ⬇️ 자동 노출 / 화이트밸런스 해제 및 수동 설정
    picam2.set_controls({
        "AwbEnable": False,
        "AeEnable": False,
        "ExposureTime": 10000,  # 마이크로초 단위 (예: 10000 = 10ms)
        "AnalogueGain": 1.0,  # 밝기 증가 (1.0~4.0)
        "ColourGains": (1.5, 1.5)  # RGB 색 보정 (G를 줄이고 색감을 맞춤)
    })

    picam2.start()
    car = CarController()
    frame_count = 0

    try:
        while True:
            print("\n[Frame captured]")
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            # 전처리
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Converted to grayscale")
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            print("Applied Gaussian blur")
            edges = cv2.Canny(blur, 30, 120)
            print("Canny edges detected")

            roi = region_of_interest(edges)
            print("ROI extracted")

            # 좌우 영역 나누기
            left_roi = roi[:, :img_center]
            right_roi = roi[:, img_center:]

            # 엣지 수 계산
            left_count = cv2.countNonZero(left_roi)
            right_count = cv2.countNonZero(right_roi)
            print(f"Left edge count: {left_count}, Right edge count: {right_count}")

            # 히스토그램 계산
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

            offset = lane_center - img_center
            print(f"Image center: {img_center}, Offset: {offset}")

            threshold = 50

            # 주행 판단 및 모터 제어
            if left_count == 0 and right_count == 0:
                print("No lane detected. Stopping.")
                direction = "stop"
                car.update(direction=None)
                car.stop_drive()

            elif abs(offset) < threshold:
                if abs(left_count - right_count) > 4000:
                    direction = "right" if left_count > right_count else "left"
                    print(f"Offset small, but unbalanced counts. Adjusting {direction}.")
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

            # 시각화
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.line(frame, (lane_center, height), (lane_center, int(height * 0.6)), (255, 0, 0), 2)
            cv2.line(frame, (img_center, height), (img_center, int(height * 0.6)), (0, 255, 0), 2)

            cv2.imshow("Lane Tracking", frame)
            cv2.imshow("ROI Edge", roi)

            # 히스토그램 시각화 (매 30프레임마다)
            if frame_count % 30 == 0:
                show_histogram(left_hist, "Left Histogram")
                show_histogram(right_hist, "Right Histogram")
            frame_count += 1

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
