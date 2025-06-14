# RC카 모드: 실시간 영상 표시 + 키 입력에 따라 조향 및 이미지 저장

import pigpio
import time
import sys
import termios
import tty
import os
import cv2
import csv
from picamera2 import Picamera2

# === 설정 ===
SERVO_PIN = 18  # 서보모터 (조향)
IN1 = 12         # DC 모터 제어
IN2 = 13
SAVE_DIR = "data"
FRAME_SAVE = True  # 이미지 저장 여부

# === 초기화 ===
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)

# 서보 각도 설정 함수 (0~180도)
def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

# DC 모터 제어 함수
def motor_forward():
    pi.write(IN1, 1)
    pi.write(IN2, 0)

def motor_backward():
    pi.write(IN1, 0)
    pi.write(IN2, 1)

def motor_stop():
    pi.write(IN1, 0)
    pi.write(IN2, 0)

# 키 입력 함수 (blocking 방식)
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# 카메라 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

# 저장 폴더 생성
if FRAME_SAVE:
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_file = open(os.path.join(SAVE_DIR, "drive_log.csv"), mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "steering_angle"])
    frame_count = 0

# 초기 상태
steering_angle = 90  # 중립
set_servo_angle(steering_angle)
motor_stop()

print("조작 키 안내:")
print(" W : 전진 | S : 후진")
print(" A : 좌회전 | D : 우회전")
print(" Q : 종료")

try:
    while True:
        frame = picam2.capture_array()
        cv2.imshow("Camera View", frame)

        key = getkey().lower()

        if key == 'w':
            motor_forward()
            print("전진")
        elif key == 's':
            motor_backward()
            print("후진")
        elif key == 'a':
            steering_angle = max(40, steering_angle - 30)
            set_servo_angle(steering_angle)
            print(f"좌회전: {steering_angle}도")
        elif key == 'd':
            steering_angle = min(180, steering_angle + 30)
            set_servo_angle(steering_angle)
            print(f"우회전: {steering_angle}도")
        elif key == 'q':
            print("종료")
            break
        else:
            motor_stop()
            print("정지")

        if FRAME_SAVE:
            filename = f"frame_{frame_count:05d}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame)
            csv_writer.writerow([filename, steering_angle])
            frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    if FRAME_SAVE:
        csv_file.close()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()
