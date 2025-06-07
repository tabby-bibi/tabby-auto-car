import cv2
import numpy as np
import time
from picamera2 import Picamera2
from car_controller import CarController

# 관심영역(ROI) 설정 함수: 이미지 하단 40% 영역만 사용
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    # ROI를 사각형으로 설정 (아래쪽만 사용)
    polygon = np.array([[
        (0, height),
        (0, int(height * 0.6)),
        (width, int(height * 0.6)),
        (width, height),
    ]], np.int32)

    # ROI 부분만 흰색(255)으로 채워 마스크 생성
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# 히스토그램 계산 함수: 각 열의 픽셀 값을 합산하여 차선 중심을 찾음
def compute_histogram(img):
    histogram = np.sum(img, axis=0)
    return histogram

def main():
    # Pi Camera 초기 설정
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    # 차량 컨트롤 객체 생성
    car = CarController()

    try:
        while True:
            # 프레임 캡처
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2  # 이미지 중심 (기준점)

            # 전처리 1: 흑백 변환 → 흐림처리 → 엣지 검출
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # ROI 마스킹 적용 (하단 영역만 분석)
            roi = region_of_interest(edges)

            # 히스토그램 생성 → 차선 중심 위치 탐색
            histogram = compute_histogram(roi)
            lane_center = np.argmax(histogram)  # 엣지가 가장 많은 열을 중심으로 추정
            offset = lane_center - img_center   # 차선 중심과 이미지 중심 간 거리 차이
            threshold = 30  # 중심 오차 허용 범위 (픽셀 단위)

            # 곡선 방향 추정: 좌/우 영역을 나누고 엣지 픽셀 수 비교
            left_roi = roi[:, :img_center]
            right_roi = roi[:, img_center:]
            left_count = cv2.countNonZero(left_roi)
            right_count = cv2.countNonZero(right_roi)

            # =======================
            # 회전 방향 판단 로직
            # =======================

            # 엣지가 전혀 없다면 → 차선 미탐지 → 정지
            if left_count == 0 and right_count == 0:
                direction = "stop"
                car.update(direction=None)
                car.stop_drive()

            # 중심선 기준으로는 직진 가능한 상태
            elif abs(offset) < threshold:
                # 단, 좌우 엣지 수 차이가 크면 → 곡선 방향 반영
                if abs(left_count - right_count) > 3000:
                    direction = "left" if left_count > right_count else "right"
                    car.update(direction=direction)
                    car.set_motor_forward()
                else:
                    # 곡선도 아니고 중심선도 정상이면 직진
                    direction = "straight"
                    car.update(direction=None)
                    car.set_motor_forward()

            # 중심선이 왼쪽으로 치우침 → 좌회전
            elif offset < -threshold:
                direction = "left"
                car.update(direction="left")
                car.set_motor_forward()

            # 중심선이 오른쪽으로 치우침 → 우회전
            else:
                direction = "right"
                car.update(direction="right")
                car.set_motor_forward()

            # =======================
            # 시각화 (화면 출력)
            # =======================

            # 방향 텍스트 출력
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 히스토그램 차선 중심선과 이미지 중심선을 시각화
            cv2.line(frame, (lane_center, height), (lane_center, int(height * 0.6)), (255, 0, 0), 2)
            cv2.line(frame, (img_center, height), (img_center, int(height * 0.6)), (0, 255, 0), 2)

            # 화면 출력 (디버그용)
            cv2.imshow("Lane Tracking", frame)
            cv2.imshow("ROI Edge", roi)

            # 종료 키: 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 예외 처리: Ctrl+C 입력 시 종료
    except KeyboardInterrupt:
        pass

    # 종료 시 모터 멈추고 카메라 정리
    finally:
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()

# 메인 함수 실행
if __name__ == "__main__":
    main()
