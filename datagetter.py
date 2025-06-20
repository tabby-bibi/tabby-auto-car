import pigpio
import time
import sys
import termios
import tty
import os
import cv2
import csv
import glob
import threading
import numpy as np
import queue
from picamera2 import Picamera2

# === 설정 ===
SERVO_PIN = 18
IN1 = 12
IN2 = 13
SAVE_DIR = "data"
FRAME_SAVE = True
motor_speed = 110  # PWM 속도 (0~255)

# === 초기화 ===
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward(speed=motor_speed):
    pi.set_PWM_dutycycle(IN1, speed)
    pi.set_PWM_dutycycle(IN2, 0)

def motor_backward(speed=motor_speed):
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, speed)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# === 카메라 초기화 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

# === 글로벌 변수 ===
latest_frame = None
running = True
key_queue = queue.Queue()

def camera_thread():
    global latest_frame, running
    while running:
        frame = picam2.capture_array()
        latest_frame = frame.copy()
        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # q는 여기서도 종료 신호로 동작
            running = False
            break

def key_input_thread():
    global running
    while running:
        k = getkey().lower()
        key_queue.put(k)
        if k == 'q':
            running = False
            break

# 카메라 스레드 시작
threading.Thread(target=camera_thread, daemon=True).start()
# 키 입력 스레드 시작
threading.Thread(target=key_input_thread, daemon=True).start()

# === 저장 폴더 및 CSV 초기화 ===
if FRAME_SAVE:
    os.makedirs(SAVE_DIR, exist_ok=True)

    existing_frames = sorted(glob.glob(os.path.join(SAVE_DIR, "frame_*.jpg")))
    if existing_frames:
        last_frame = existing_frames[-1]
        frame_count = int(os.path.basename(last_frame).split("_")[1].split(".")[0]) + 1
    else:
        frame_count = 0

    csv_path = os.path.join(SAVE_DIR, "drive_log.csv")
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(["frame", "steering_angle", "label", "center_x", "error"])

# === 초기 상태 ===
steering_angle = 90
set_servo_angle(steering_angle)
motor_stop()

print("조작 키 안내:")
print(" W : 전진 | S : 후진")
print(" A : 좌회전 | D : 우회전")
print(" 스페이스바 : 정지 상태 저장")
print(" Q : 종료")

try:
    while running:
        if not key_queue.empty():
            key = key_queue.get()
        else:
            key = None

        if key is None:
            time.sleep(0.01)
            continue

        saving_frames = False
        label = "stop"  # 기본은 정지

        if key == 'w':
            print("전진")
            label = "center"
            saving_frames = True
        elif key == 's':
            print("후진")
            label = "backward"
            saving_frames = True
        elif key == 'a':
            steering_angle = max(10, steering_angle - 30)
            set_servo_angle(steering_angle)
            print(f"좌회전: {steering_angle}도")
            label = "left"
            saving_frames = True
        elif key == 'd':
            steering_angle = min(150, steering_angle + 30)
            set_servo_angle(steering_angle)
            print(f"우회전: {steering_angle}도")
            label = "right"
            saving_frames = True
        elif key == ' ':
            motor_stop()
            print("정지 상태 저장")
            label = "stop"
            saving_frames = True
        elif key == 'q':
            print("종료")
            running = False
            break
        else:
            motor_stop()
            continue

        if FRAME_SAVE and latest_frame is not None and saving_frames:
            frame = latest_frame.copy()

            # ROI 설정 (하단 1/4 영역)
            roi = frame[360:480, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            histogram = np.sum(edges, axis=0)
            midpoint = histogram.shape[0] // 2

            leftx = np.argmax(histogram[:midpoint])
            rightx = np.argmax(histogram[midpoint:]) + midpoint
            center_x = (leftx + rightx) // 2
            frame_center = frame.shape[1] // 2
            error = center_x - frame_center

            # 시각화
            annotated = frame.copy()
            cv2.line(annotated, (center_x, 460), (center_x, 480), (255, 0, 0), 3)   # 차선 중심 파란색 선
            cv2.line(annotated, (frame_center, 460), (frame_center, 480), (0, 0, 255), 3)  # 화면 중앙 빨간색 선
            cv2.putText(annotated, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

            filename = f"frame_{frame_count:05d}_{label}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, annotated)

            csv_writer.writerow([filename, steering_angle, label, center_x, error])
            csv_file.flush()
            print(f"저장: {filename} - 각도: {steering_angle} - 라벨: {label} - 중심점: {center_x} - 오차: {error}")
            frame_count += 1

except KeyboardInterrupt:
    print("강제 종료됨 (Ctrl+C)")

finally:
    running = False
    time.sleep(0.5)

    print("=== 종료 정리 중 ===")
    if FRAME_SAVE:
        csv_file.close()

    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("모든 자원 해제 완료. 안전 종료되었습니다.")
