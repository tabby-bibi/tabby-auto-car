### 주행이 미숙한 코드 곡선을 정확히 읽지 못함 ###




# 필요한 라이브러리 불러오기
import cv2  # OpenCV - 이미지 처리
import numpy as np  # 행렬 계산용
import time  # 시간 측정
from picamera2 import Picamera2  # Pi Camera 2 사용
from car_controller import CarController  # 사용자 정의 모터 제어 클래스 (별도 작성된 클래스라고 가정)

# ROI 마스크를 위한 설정값 (하단, 상단 너비와 높이 비율)
POLY_BOTTOM_WIDTH = 2.0
POLY_TOP_WIDTH = 0.4
POLY_HEIGHT = 0.3

# 흰색 계열만 필터링 (차선을 강조하기 위함)
def filter_white(img):
    lower = np.array([150, 150, 150])  # 밝은 회색 이상
    upper = np.array([255, 255, 255])  # 완전한 흰색
    mask = cv2.inRange(img, lower, upper)
    return cv2.bitwise_and(img, img, mask=mask)

# 관심 영역(ROI) 설정: 차선이 있을 법한 전방 바닥 쪽만 남김
def region_of_interest(img):
    height, width = img.shape[:]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(width * (1 - POLY_BOTTOM_WIDTH) / 2), height),
        (int(width * (1 - POLY_TOP_WIDTH) / 2), int(height * (1 - POLY_HEIGHT))),
        (int(width * (1 + POLY_TOP_WIDTH) / 2), int(height * (1 - POLY_HEIGHT))),
        (int(width * (1 + POLY_BOTTOM_WIDTH) / 2), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# 직선 라인을 좌/우 차선으로 분리
def separate_lines(lines, img_center):
    left_lines = []
    right_lines = []
    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 수직선 제거 (x 변화 없음)
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)  # 기울기 계산

        if abs(slope) < 0.3:  # 거의 수평선은 제외 (노이즈 가능성 큼)
            continue

        # 화면의 왼쪽에 있는 기울기가 음수인 라인은 왼쪽 차선
        if slope < 0 and x1 < img_center and x2 < img_center:
            left_lines.extend([[x1, y1], [x2, y2]])
        # 화면의 오른쪽에 있는 기울기가 양수인 라인은 오른쪽 차선
        elif slope > 0 and x1 > img_center and x2 > img_center:
            right_lines.extend([[x1, y1], [x2, y2]])

    return left_lines, right_lines

# 차선 포인트를 2차 방정식으로 근사
def fit_polynomial(points):
    if len(points) < 3:
        return None
    points = np.array(points)
    fit = np.polyfit(points[:,1], points[:,0], 2)  # y를 기준으로 x를 피팅
    return fit

# 피팅된 곡선을 포인트로 변환
def make_polynomial_points(fit, y1, y2, n=50):
    if fit is None:
        return None
    ys = np.linspace(y1, y2, n)
    xs = fit[0]*ys**2 + fit[1]*ys + fit[2]
    points = np.array([np.vstack((xs, ys)).T], dtype=np.int32)
    return points

# 메인 실행 함수
def main():
    # 카메라 및 모터 객체 초기화
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    car = CarController()

    last_full_lane_time = time.time()  # 마지막으로 좌/우 차선을 모두 본 시점

    try:
        while True:
            # 카메라로 한 프레임 캡처
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            # 1. 색상 필터링으로 흰색 차선 강조
            filtered = filter_white(frame)

            # 2. 흑백 변환 → 블러 처리 → Canny 엣지 검출
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 100)

            # 3. 관심 영역 마스킹
            masked = region_of_interest(edges)

            # 4. Hough Line 변환으로 직선 추출
            lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=30)

            # 5. 좌/우 차선 나누기
            left_points, right_points = separate_lines(lines, img_center)

            # 6. 각각 2차곡선으로 피팅
            left_fit = fit_polynomial(left_points)
            right_fit = fit_polynomial(right_points)

            y1 = height
            y2 = int(height * 0.6)  # 곡선 상단 y 지점

            # 7. 좌/우 차선을 모두 인식한 경우
            if left_fit is not None and right_fit is not None:
                last_full_lane_time = time.time()  # 시간 업데이트

                # 차선을 따라 시각화용 포인트 생성
                left_curve = make_polynomial_points(left_fit, y2, y1)
                right_curve = make_polynomial_points(right_fit, y2, y1)

                if left_curve is not None and right_curve is not None:
                    # 차선 사이 영역 색칠
                    pts = np.vstack((left_curve[0], right_curve[0][::-1])).astype(np.int32)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                    # 차선 그리기
                    cv2.polylines(frame, [left_curve], isClosed=False, color=(0, 255, 255), thickness=5)
                    cv2.polylines(frame, [right_curve], isClosed=False, color=(0, 255, 255), thickness=5)

                    # 차선 중심 계산
                    center_xs = (left_curve[0][:,0] + right_curve[0][:,0]) // 2
                    mid_index = len(center_xs) // 2
                    center_offset = center_xs[mid_index] - img_center

                    # 중심 위치에 따른 방향 결정
                    if abs(center_offset) < 20:
                        direction = "straight"
                    elif center_offset > 20:
                        direction = "right"
                    else:
                        direction = "left"

                    # 방향에 따라 조향 및 구동
                    if direction == "straight":
                        car.update(direction=None)
                        car.set_motor_forward()
                    elif direction == "left":
                        car.update(direction="left")
                    elif direction == "right":
                        car.update(direction="right")
                    else:
                        car.update(direction=None)
                        car.stop_drive()

                    # 화면에 방향 표시
                    cv2.putText(frame, f"Direction: {direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                else:
                    car.stop_drive()
                    cv2.putText(frame, "No lane detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 8. 한쪽 차선만 인식된 경우 (보정 운전 시도)
            elif left_fit is not None or right_fit is not None:
                # 최근 좌우 차선 모두 인식된 시점이 2초 이내면 보정 운전
                if time.time() - last_full_lane_time < 2.0:
                    direction = "right" if left_fit is not None else "left"
                    car.update(direction=direction)
                    car.set_motor_forward()
                    cv2.putText(frame, f"Partial lane detected, direction: {direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    car.update(direction=None)
                    car.stop_drive()
                    cv2.putText(frame, "No full lane for > 2s, stopped", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 9. 차선이 전혀 감지되지 않은 경우
            else:
                car.update(direction=None)
                car.stop_drive()
                cv2.putText(frame, "No lane detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 화면 출력
            cv2.imshow("Lane Detection", frame)
            cv2.imshow("Edges", edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 안전하게 정지 및 자원 정리
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()

# 프로그램 시작점
if __name__ == "__main__":
    main()
