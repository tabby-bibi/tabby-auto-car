# 주행으로 데이터 수집하는 코드

import cv2
import numpy as np
import pigpio
import csv
import time
import os
import sys
import termios
import tty
import select
from picamera2 import Picamera2
from threading import Thread, Lock

# 서보 & 모터 제어 핀 정의
SERVO_PIN = 18
IN1 = 12
IN2 = 13

# pigpio 초기화
pi = pigpio.pi()

# 서보 각도 설정 함수
def set_servo_angle(angle):
    pulsewidth = int(500 + (angle / 180.0) * 2000)
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward():
    pi.write(IN1, 1)
    pi.write(IN2, 0)

def motor_backward():
    pi.write(IN1, 0)
    pi.write(IN2, 1)

def motor_stop():
    pi.write(IN1, 0)
    pi.write(IN2, 0)

# 키보드 입력 처리
def getkey():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == '[':
                if ch3 == 'A': return 'up'
                elif ch3 == 'B': return 'down'
                elif ch3 == 'C': return 'right'
                elif ch3 == 'D': return 'left'
        return ch1.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# 저장 디렉토리 생성
os.makedirs('data', exist_ok=True)

# PiCamera2 초기화
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": (320, 240)}))
picam.start()
time.sleep(1)  # 카메라 워밍업

# CSV 저장
csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

frame_count = 0
steering_angle = 90
throttle = 0
latest_frame = None
frame_lock = Lock()

# 카메라 스레드: 최신 프레임 계속 받아오기
def camera_loop():
    global latest_frame
    while True:
        frame = picam.capture_array()
        with frame_lock:
            latest_frame = frame

cam_thread = Thread(target=camera_loop, daemon=True)
cam_thread.start()

try:
    print("↑: 전진, ↓: 후진, ←: 좌회전, →: 우회전, Q: 종료")

    while True:
        # 최신 프레임 가져오기
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # 키 입력 처리
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = getkey()
            if key == 'up':
                steering_angle = 90
                set_servo_angle(steering_angle)
                throttle = 40
                motor_forward()
            elif key == 'down':
                throttle = -40
                motor_backward()
            elif key == 'left':
                steering_angle = max(0, steering_angle - 10)
                set_servo_angle(steering_angle)
            elif key == 'right':
                steering_angle = min(180, steering_angle + 10)
                set_servo_angle(steering_angle)
            elif key == 'q':
                break
            else:
                throttle = 0
                motor_stop()

        # 방향 문자열
        if steering_angle < 80:
            direction = "left"
        elif steering_angle > 100:
            direction = "right"
        else:
            direction = "straight"

        # 이미지 저장
        filename = f"data/frame_{frame_count:05d}_{direction}.jpg"
        cv2.imwrite(filename, frame)

        # CSV 저장
        csv_writer.writerow([frame_count, steering_angle, throttle])
        frame_count += 1

        # 화면 표시
        cv2.imshow("Driving", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam.stop()
    cv2.destroyAllWindows()
