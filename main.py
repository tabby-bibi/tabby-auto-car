# OpenCV와 NumPy는 영상 처리에 사용됨
import cv2
import numpy as np
import time

# Raspberry Pi Camera 2용 라이브러리
from picamera2 import Picamera2

# 사용자 정의 차량 제어 클래스 (앞서 작성한 car_controller.py에 포함)
from car_controller import CarController

# 관심 영역 설정 비율 (차선 검출에 사용할 하단 다각형 ROI 범위 설정용)
POLY_BOTTOM_WIDTH = 1.5  # 이미지 하단의 ROI 너비 비율
POLY_TOP_WIDTH = 0.3     # 이미지 상단의 ROI 너비 비율
POLY_HEIGHT = 0.6        # 이미지에서 ROI의 높이 비율

# 흰색 필터링 함수 (차선이 흰색일 경우 유용)
def filter_white(img):
    """흰색 영역만 추출하는 필터링 함수"""
    lower = np.array([160, 160, 160])  # BGR 흰색 하한
    upper = np.array([255, 255, 255])  # BGR 흰색 상한
    mask = cv2.inRange(img, lower, upper)  # 범위에 해당하는 부분만 추출
    return cv2.bitwise_and(img, img, mask=mask)  # 원본 이미지에서 흰색 부분만 남김

# 관심 영역(ROI)을 적용하여 하단 차선 부분만 남기는 함수
def region_of_interest(img):
    height, width = img.shape[:2]  # 이미지 높이와 너비 가져오기
    mask = np.zeros_like(img)  # 마스크는 그레이스케일 형식 (초기값 0)

    # ROI 다각형 좌표 계산
    polygon = np.array([[  
        (int(width * (1 - POLY_BOTTOM_WIDTH) / 2), height),
        (int(width * (1 - POLY_TOP_WIDTH) / 2), int(height * (1 - POLY_HEIGHT))),
        (int(width * (1 + POLY_TOP_WIDTH) / 2), int(height * (1 - POLY_HEIGHT))),
        (int(width * (1 + POLY_BOTTOM_WIDTH) / 2), height)
    ]], np.int32)

    # ROI 영역 마스크에 흰색 채우기
    cv2.fillPoly(mask, polygon, 255)
    masked_img = cv2.bitwise_and(img,mask)  # ROI 바깥은 제거
    return masked_img

# 허프 직선 결과를 좌우 차선으로 분리하는 함수
def separate_lines(lines, img_center):
    left_lines = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:  # 기울기가 무한인 수직선 제거
            continue
        slope = (y2 - y1) / (x2 - x1)  # 기울기 계산

        if abs(slope) < 0.3:  # 기울기가 거의 없는 선(수평선)은 무시
            continue

        # 좌측 차선: 음의 기울기이며 이미지 중심 왼쪽에 위치
        if slope < 0 and x1 < img_center and x2 < img_center:
            left_lines.append([x1, y1])
            left_lines.append([x2, y2])
        # 우측 차선: 양의 기울기이며 이미지 중심 오른쪽에 위치
        elif slope > 0 and x1 > img_center and x2 > img_center:
            right_lines.append([x1, y1])
            right_lines.append([x2, y2])
    return left_lines, right_lines

# 점들로부터 선형 회귀하여 직선의 기울기와 y절편을 구함
def fit_line(points):
    if len(points) == 0:
        return None
    points = np.array(points)
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = vy / vx
    intercept = y - slope * x
    return slope[0], intercept[0]

# 기울기와 절편으로 실제 두 점을 만들어 직선을 시각화
def make_line_points(y1, y2, slope, intercept):
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

# 차선 중앙이 이미지 중앙보다 좌/우에 있는지에 따라 방향 결정
def decide_direction(center_lane_x, img_center,left_slope,right_slope,
position_thresh=30, slope_thresh=0.6,slope_diff_thresh=0.5):
    
    diff = center_lane_x - img_center
    
    if abs(diff) > position_thresh and abs(left_slope) > slope_thresh and abs(right_slope) > slope_thresh and abs(left_slope - right_slope) > slope_diff_thresh:
  
        return "right" if diff > 0 else "left"
    return "straight"
    
# 메인 함수: 카메라로 영상 받아서 차선 인식 및 차량 제어
def main():
    # 카메라 설정
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    # 차량 제어 인스턴스 생성
    car = CarController()

    try:
        while True:
            frame = picam2.capture_array()  # 프레임 캡처

            height, width = frame.shape[:2]
            img_center = width // 2

            filtered = filter_white(frame)  # 흰색 차선 필터링
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
            edges = cv2.Canny(gray, 50, 120)  # 엣지 검출
            masked = region_of_interest(edges)  # ROI 마스킹

            # 허프 직선 변환으로 차선 후보 찾기
            lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=20)

            # 차선 좌/우 분리
            left_points, right_points = separate_lines(lines, img_center)

            # 각각 선형 회귀
            left_fit = fit_line(left_points)
            right_fit = fit_line(right_points)

            y1 = height
            y2 = int(height * 0.6)

            if left_fit is not None and right_fit is not None:
                # 양쪽 차선이 검출된 경우
                left_slope, left_intercept = left_fit
                right_slope, right_intercept = right_fit

                left_line = make_line_points(y1, y2, left_slope, left_intercept)
                right_line = make_line_points(y1, y2, right_slope, right_intercept)

                if left_line and right_line:
                    # 양쪽 직선에서 중심 x 좌표 계산
                    center_lane_x = (left_line[0][0] + right_line[0][0]) // 2

                    # 차선 영역 시각화 (초록색 투명 다각형)
                    pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]])
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                    # 실제 차선 선 그리기
                    cv2.line(frame, left_line[0], left_line[1], (0, 255, 255), 5)
                    cv2.line(frame, right_line[0], right_line[1], (0, 255, 255), 5)

                    # 주행 방향 결정
                    direction = decide_direction(center_lane_x, img_center,left_slope,right_slope)

                    if direction == "straight":
                        car.update(direction=None)
                        car.set_motor_forward()         # 방향 조향 정지
                    if direction == "right":
                        car.update(direction="right")
                    if direction == "left":
                        car.update(direction="left")
                    else:
                        car.update(direction=None)
                                                        # 완전정지

                    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                else:
                    # 차선을 인식하지 못한 경우
                    car.stop_drive()
                    cv2.putText(frame, "No lane detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                # 양쪽 차선 중 하나라도 검출 실패 시
                car.stop_drive()
                cv2.putText(frame, "No lane detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # 결과 영상 출력
            cv2.imshow("Lane Detection", frame)
            
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        car.stop_drive()  # 종료 시 차량 정지
        picam2.stop()     # 카메라 종료
        cv2.destroyAllWindows()  # 창 닫기

# 프로그램 시작점
if __name__ == "__main__":
    main()
