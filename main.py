import cv2
import numpy as np
from picamera2 import Picamera2
import time
from car_controller import CarController

# ================================
# 영상에서 흰색 테이프만 필터링
# ================================
def filter_white_tape(img):
    """
    BGR 이미지에서 밝은 흰색 영역만 필터링합니다.
    흰색 테이프만 검출하기 위함.
    """
    lower_white = np.array([200, 200, 200])  # BGR 최소값
    upper_white = np.array([255, 255, 255])  # BGR 최대값

    # 지정한 범위 내 픽셀을 흰색(255), 나머지는 검정(0)으로 마스크 생성
    mask = cv2.inRange(img, lower_white, upper_white)
    # 원본 이미지에서 흰색 영역만 추출
    filtered = cv2.bitwise_and(img, img, mask=mask)
    return filtered

# ==================================
# 엣지 검출 및 관심 영역 마스킹 함수
# ==================================
def region_of_interest(img):
    """
    입력된 이미지에서 관심 영역(도로 영역)만 추출하는 함수
    차선이 위치할 가능성이 높은 삼각형 영역만 검출한다.
    """
    height, width = img.shape[:2]
    mask = np.zeros_like(img)  # 입력 이미지와 같은 크기의 검정 마스크 생성

    # 삼각형 모양의 관심 영역 꼭지점 좌표
    polygon = np.array([
        [(int(width * 0.1), height),          # 왼쪽 아래 모서리
         (int(width * 0.45), int(height * 0.6)),  # 왼쪽 위 쪽
         (int(width * 0.55), int(height * 0.6)),  # 오른쪽 위 쪽
         (int(width * 0.9), height)]          # 오른쪽 아래 모서리
    ])

    # 마스크에 흰색으로 관심 영역 채우기
    cv2.fillPoly(mask, polygon, 255)

    # 입력 영상과 마스크를 bitwise_and 연산하여 관심 영역만 추출
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# =======================
# 허프 변환으로 직선 검출
# =======================
def hough_lines(img):
    """
    엣지 이미지에서 허프 변환으로 직선을 검출한다.
    """
    # rho, theta, threshold, minLineLength, maxLineGap 조절 필요
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=100)
    return lines

# ===============================
# 직선을 좌우 차선으로 분리하는 함수
# ===============================
def separate_lines(lines, img_width):
    """
    검출된 모든 직선을 좌우로 분리한다.
    기울기가 양수면 우측, 음수면 좌측 차선으로 판단.
    """
    left_lines = []
    right_lines = []
    slope_threshold = 0.3  # 너무 수평인 선은 제외

    if lines is None:
        return left_lines, right_lines

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:  # 기울기 계산 중 0 나누기 방지
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < slope_threshold:
                continue

            if slope < 0 and x1 < img_width // 2 and x2 < img_width // 2:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > img_width // 2 and x2 > img_width // 2:
                right_lines.append((x1, y1, x2, y2))
    return left_lines, right_lines

# =======================================
# 여러 직선을 하나의 직선으로 회귀선 산출하는 함수
# =======================================
def fit_line(lines):
    """
    여러 점들을 모아 선형 회귀를 하여
    하나의 대표 직선(기울기, 절편)을 계산한다.
    """
    if len(lines) == 0:
        return None

    x_coords = []
    y_coords = []

    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    # np.polyfit으로 1차 다항식(직선) 피팅
    fit = np.polyfit(x_coords, y_coords, 1)  # y = m*x + b
    slope = fit[0]
    intercept = fit[1]
    return slope, intercept

# =======================
# 회귀선 좌표 계산 함수
# =======================
def make_line_points(y1, y2, slope, intercept):
    """
    y축 시작과 끝 위치와
    회귀선의 기울기, 절편으로
    x축 시작과 끝 좌표 계산
    """
    if slope == 0:  # 기울기 0 방지
        x1 = x2 = 0
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    return (x1, y1, x2, y2)

# ===================
# 진행 방향 예측 함수
# ===================
def predict_direction(left_line, right_line, img_width):
    """
    좌우 차선의 교차점 x 좌표를 계산하여
    중심과 비교해 좌회전, 우회전, 직진 판단
    """
    if left_line is None or right_line is None:
        return "stop"

    left_slope, left_intercept = left_line
    right_slope, right_intercept = right_line

    # 두 직선의 교차점 x 좌표 공식
    x_intersect = (right_intercept - left_intercept) / (left_slope - right_slope)

    center = img_width / 2
    threshold = 20  # 중심 허용 범위

    if x_intersect < center - threshold:
        return "left"
    elif x_intersect > center + threshold:
        return "right"
    else:
        return "straight"

# ========================
# 차선 선 그리기 함수
# ========================
def draw_lines(img, left_line, right_line):
    """
    영상에 좌우 차선을 그리고,
    차선 사이를 반투명으로 채운다.
    """
    line_img = np.zeros_like(img)

    height = img.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    if left_line is not None:
        left_pts = make_line_points(y1, y2, left_line[0], left_line[1])
        cv2.line(line_img, (left_pts[0], left_pts[1]), (left_pts[2], left_pts[3]), (0, 255, 255), 5)

    if right_line is not None:
        right_pts = make_line_points(y1, y2, right_line[0], right_line[1])
        cv2.line(line_img, (right_pts[0], right_pts[1]), (right_pts[2], right_pts[3]), (0, 255, 255), 5)

    # 좌우 차선 사이를 다각형으로 채우기 (반투명)
    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_pts[0], left_pts[1]],
            [left_pts[2], left_pts[3]],
            [right_pts[2], right_pts[3]],
            [right_pts[0], right_pts[1]]
        ])
        cv2.fillPoly(line_img, [pts], (0, 230, 30))

    # 원본 영상과 합성
    combined = cv2.addWeighted(img, 0.7, line_img, 0.3, 0)
    return combined

# ========================
# 메인 함수
# ========================
def main():
    # PiCamera2 객체 생성 및 설정
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    # 차 컨트롤러 객체 생성
    car = CarController()

    try:
        while True:
            # 프레임 캡처
            frame = picam2.capture_array()

            # 흰색 테이프 필터링
            filtered = filter_white_tape(frame)

            # 그레이스케일 변환 (엣지 검출 전 단계)
            gray = cv2.cvtColor(filtered,
