# RC카를 직접 운전하여 이미지 데이터를 저장하는 코드


import os
import cv2
import numpy as np
import pigpio
import csv
import time
import sys
import termios
import tty
import select
from picamera2 import Picamera2

# 서보 & 모터 제어 핀 정의
SERVO_PIN = 18
IN1 = 12
IN2 = 13

# pigpio 초기화
pi = pigpio.pi()

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

def getkey():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Picamera2 설정
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()
time.sleep(2)

# 디렉토리 생성
os.makedirs("data", exist_ok=True)

# CSV 저장 준비
csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

frame_count = 0
steering_angle = 90
throttle = 0

try:
    print("W: 전진, S: 후진, A: 좌회전, D: 우회전, Q: 종료")

    while True:
        frame = picam2.capture_array()

        key = None
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = getkey().lower()

        if key == 'w':
            throttle = 1
            motor_forward()
        elif key == 's':
            throttle = -1
            motor_backward()
        elif key == 'a':
            steering_angle = max(0, steering_angle - 5)
            set_servo_angle(steering_angle)
        elif key == 'd':
            steering_angle = min(180, steering_angle + 5)
            set_servo_angle(steering_angle)
        elif key == 'q':
            break
        else:
            throttle = 0
            motor_stop()

        # 프레임 저장
        filename = f'data/frame_{frame_count:05d}.jpg'
        cv2.imwrite(filename, frame)

        # CSV 기록
        csv_writer.writerow([frame_count, steering_angle, throttle])
        frame_count += 1

        # 화면에 표시 (선택사항)
        cv2.imshow("Driving", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()

