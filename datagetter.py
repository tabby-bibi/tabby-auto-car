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
from picamera2 import Picamera2

# === 설정 ===
SERVO_PIN = 18
IN1 = 12
IN2 = 13
SAVE_DIR = "data"
FRAME_SAVE = True
motor_speed = 110
ANGLE_STEP = 20
STEERING_CENTER = 90

# === 초기화 ===
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)
pi.set_servo_pulsewidth(SERVO_PIN, 0)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

# ✅ 블로킹 방식 키 입력
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# === 카메라 설정 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

# === 프레임 캡처 쓰레드 ===
latest_frame = None
running = True

def camera_thread():
    global latest_frame
    while running:
        latest_frame = picam2.capture_array()
        cv2.imshow("Camera View", latest_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

threading.Thread(target=camera_thread, daemon=True).start()

# === CSV 파일 초기화 ===
os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, "drive_log.csv")
file_exists = os.path.exists(csv_path)
csv_file = open(csv_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow(["frame", "steering_angle", "label"])

existing_frames = sorted(glob.glob(os.path.join(SAVE_DIR, "frame_*_*.jpg")))
if existing_frames:
    last_frame = os.path.basename(existing_frames[-1])
    frame_count = int(last_frame.split("_")[1])
else:
    frame_count = 0

# === 초기값 ===
steering_angle = STEERING_CENTER
set_servo_angle(steering_angle)
motor_stop()

print("조작 키 안내: W(전진) | A(좌) | D(우) | 스페이스(정지 저장) | Q(종료)")

# === 메인 루프 ===
try:
    while running:
        key = getkey().lower()
        label = None
        save_frame = False

        if key == 'w':
            label = "straight"
            save_frame = True

        elif key == 'a':
            steering_angle = max(0, steering_angle - ANGLE_STEP)
            set_servo_angle(steering_angle)
            label = "left"
            save_frame = True
            time.sleep(0.2)
            steering_angle = STEERING_CENTER
            set_servo_angle(steering_angle)

        elif key == 'd':
            steering_angle = min(180, steering_angle + ANGLE_STEP)
            set_servo_angle(steering_angle)
            label = "right"
            save_frame = True
            time.sleep(0.2)
            steering_angle = STEERING_CENTER
            set_servo_angle(steering_angle)

        elif key == ' ':
            label = "stop"
            save_frame = True

        elif key == 'q':
            print("종료됨.")
            break

        if save_frame and latest_frame is not None:
            annotated = latest_frame.copy()
            text = f"{label}, angle={steering_angle}"
            cv2.putText(annotated, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
