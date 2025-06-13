# OpenCV, numpy, PiCamera2 라이브러리 임포트
import cv2
import numpy as np
from picamera2 import Picamera2
import time

# 이미지 전처리 함수: 그레이스케일, 블러, Canny 엣지 검출, ROI 설정
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 컬러 영상을 흑백으로 변환
    blur = cv2.GaussianBlur(gray, (5, 5), 0)        # 노이즈 제거를 위한 가우시안 블러
    edges = cv2.Canny(blur, 50, 150)                # 엣지 검출 (경계선 강조)

    height, width = edges.shape                     # 영상의 세로(height), 가로(width) 크기 얻기
    mask = np.zeros_like(edges)                     # ROI 마스크 초기화
    polygon = np.array([[
        (0, height),
        (0, height // 2),
        (width, height // 2),
        (width, height)
    ]], np.int32)                                    # 화면 하단 절반만 사용하는 ROI 영역 정의
    cv2.fillPoly(mask, polygon, 255)                # 다각형 영역을 흰색(255)으로 채움
    roi = cv2.bitwise_and(edges, mask)              # ROI 이외 영역 제거 (AND 연산)
    return roi

# 슬라이딩 윈도우 방식으로 왼쪽/오른쪽 차선 추적
def sliding_window(binary):
    # 영상 하단 절반의 픽셀 수를 수평 방향으로 누적합 → 히스토그램
    hist = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = len(hist) // 2
    leftx_base = np.argmax(hist[:midpoint])         # 왼쪽 차선 시작 위치
    rightx_base = np.argmax(hist[midpoint:]) + midpoint  # 오른쪽 차선 시작 위치

    # 윈도우 수, 윈도우 너비, 최소 픽셀 수 설정
    nwindows = 9
    margin = 50
    minpix = 50
    window_height = binary.shape[0] // nwindows     # 각 윈도우 높이

    # 흰 픽셀 좌표 추출
    nonzero = binary.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]

    # 윈도우의 현재 위치 초기화
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 픽셀 인덱스를 저장할 리스트
    left_lane_inds = []
    right_lane_inds = []

    # 슬라이딩 윈도우를 위에서 아래로 반복
    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 현재 윈도우 안에 포함된 흰색 픽셀 찾기
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # 찾은 픽셀을 리스트에 추가
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 픽셀이 충분히 있다면, 윈도우 중심을 이동
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # 모든 윈도우에서 찾은 인덱스 병합
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 인덱스를 기반으로 좌표 추출
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 양쪽 차선을 모두 찾은 경우에만 폴리핏 수행
    if len(leftx) > 0 and len(rightx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)       # 2차 곡선 피팅 (왼쪽 차선)
        right_fit = np.polyfit(righty, rightx, 2)    # 오른쪽 차선

        ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])  # 세로 범위
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]   # 곡선 계산
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        center_fitx = (left_fitx + right_fitx) / 2     # 중앙 차선 계산

        return ploty, left_fitx, right_fitx, center_fitx
    else:
        # 차선을 제대로 못 찾은 경우
        return None, None, None, None

# ------- PiCamera2 초기화 -------
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)  # 카메라 부팅 대기

# --------- 메인 루프 ----------
while True:
    frame = picam2.capture_array()       # PiCamera2에서 프레임 가져오기
    binary = preprocess(frame)           # 전처리 → 이진 이미지 (Canny + ROI)
    ploty, left_fitx, right_fitx, center_fitx = sliding_window(binary)  # 슬라이딩 윈도우 수행

    result = frame.copy()                # 결과 시각화를 위한 복사본

    if center_fitx is not None:         # 차선이 감지되었다면
        for i in range(len(ploty)):
            # 왼쪽 차선: 파란색
            cv2.circle(result, (int(left_fitx[i]), int(ploty[i])), 2, (255, 0, 0), -1)
            # 오른쪽 차선: 빨간색
            cv2.circle(result, (int(right_fitx[i]), int(ploty[i])), 2, (0, 0, 255), -1)
            # 중앙 차선: 초록색
            cv2.circle(result, (int(center_fitx[i]), int(ploty[i])), 2, (0, 255, 0), -1)

        # 중앙 차선의 맨 아래 위치와 화면 중심의 차이 계산 (조향 판단 기준)
        lane_center = center_fitx[-1]
        frame_center = frame.shape[1] // 2
        offset = frame_center - lane_center

        # 오프셋 값을 영상에 출력
        cv2.putText(result, f"Offset: {offset:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 결과 영상 출력
    cv2.imshow("Lane Detection Realtime", result)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

# 종료 처리
cv2.destroyAllWindows()
picam2.stop()