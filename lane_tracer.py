import cv2
import numpy as np


def process_frame(frame):
    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러로 노이즈 제거
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출
    edges = cv2.Canny(blur, 50, 150)
    #cv2.imshow("Canny edge", edges)
    # 관심영역 설정 (하단 부분만)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi, 255)
    cropped = cv2.bitwise_and(edges, mask)

    return cropped


def find_lane_center(frame):
    height, width = frame.shape
    histogram = np.sum(frame[height // 2:, :], axis=0)

    midpoint = width // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    lane_center = (left_base + right_base) // 2

    return lane_center, width // 2  # 차선 중심, 화면 중심

def calculate_steering_angle(lane_center, frame_center):
    offset = lane_center - frame_center
    max_angle = 30
    max_offset = frame_center

    angle = (offset / max_offset) * max_angle
    servo_angle = 90 + angle  # 정중앙은 90도

    # 각도 클램핑
    servo_angle = max(60, min(120, servo_angle))  # 필요 시 조정

    return servo_angle

def control_car(lane_center, frame_center):
    if lane_center < frame_center - 20:
        print("좌회전")
        return "left"
    elif lane_center > frame_center + 20:
        print("우회전")
        return "right"
    else:
        print("직진")
        return "straight"


cap = cv2.VideoCapture(0)  # 라즈베리파이 카메라 사용 시 적절히 조정

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = process_frame(frame)
    lane_center, frame_center = find_lane_center(processed)
    control_car(lane_center, frame_center)
    servo_angle = calculate_steering_angle(lane_center, frame_center)
    
    
    # 디버깅용 영상 표시
    cv2.line(frame, (lane_center, 0), (lane_center, frame.shape[0]), (0, 255, 0), 2)
    cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (255, 0, 0), 1)
    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

