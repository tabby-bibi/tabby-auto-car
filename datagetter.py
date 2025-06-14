import cv2
import numpy as np
import pigpio
import csv
import time
from threading import Thread
from picamera2 import Picamera2
import sys
import termios
import tty

# 서보 & 모터 제어 핀 정의
SERVO_PIN = 18
IN1 = 17
IN2 = 27

# pigpio 초기화
pi = pigpio.pi()

def set_servo_angle(angle):
    pulsewidth = int(500 + (angle / 180.0) * 2000)  # 500~2500us 펄스폭
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
        ch = sys.stdin.read(1)  # blocking 호출
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# PiCamera2 초기화 및 설정
picam = Picamera2()
preview_config = picam.create_preview_configuration(main={"size": (320, 240)})
picam.configure(preview_config)
picam.start()
time.sleep(1)  # 카메라 워밍업

# 저장 준비
import os
os.makedirs('data', exist_ok=True)

csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

frame_count = 0
steering_angle = 90  # 중립
throttle = 0         # 0: 정지, 1: 전진, -1: 후진

print("W: 전진, S: 후진, A: 좌회전, D: 우회전, Q: 종료")

try:
    while True:
        # 프레임 캡처
        frame = picam.capture_array()

        # 키 입력 (blocking)
        key = getkey().lower()

        # 키에 따른 모터/서보 제어
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

        # 영상 저장
        cv2.imwrite(f'data/frame_{frame_count:05d}.jpg', frame)

        # CSV 기록
        csv_writer.writerow([frame_count, steering_angle, throttle])

        frame_count += 1

        # 화면 표시
        cv2.imshow('Driving', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    picam.stop()
    cv2.destroyAllWindows()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
