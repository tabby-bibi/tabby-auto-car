# 주행으로 데이터 수집하는 코드

import pigpio
import time
import sys
import termios
import tty
import os
import cv2
from picamera2 import Picamera2

# 핀 번호 설정
SERVO_PIN = 18
IN1 = 12
IN2 = 13

# pigpio 초기화
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

# 키보드 입력 받는 함수 (blocking)
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()

# 저장할 디렉토리 생성
os.makedirs('data', exist_ok=True)

# PiCamera2 초기화 및 시작
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": (320, 240)}))
picam.start()
time.sleep(1)  # 워밍업

# 초기 상태
steering_angle = 90
set_servo_angle(steering_angle)
motor_stop()

frame_count = 0

print("조작 키 안내:")
print(" W : 전진")
print(" S : 후진")
print(" A : 좌회전")
print(" D : 우회전")
print(" Q : 종료")

try:
    while True:
        key = getkey()

        # 카메라 이미지 캡처
        frame = picam.capture_array()

        if key == 'w':
            motor_forward()
            throttle = 40
            direction = "straight"
            print("전진")
        elif key == 's':
            motor_backward()
            throttle = -40
            direction = "straight"
            print("후진")
        elif key == 'a':
            steering_angle = max(40, steering_angle - 10)
            set_servo_angle(steering_angle)
            throttle = 0
            direction = "left"
            print(f"좌회전: {steering_angle}도")
        elif key == 'd':
            steering_angle = min(140, steering_angle + 10)
            set_servo_angle(steering_angle)
            throttle = 0
            direction = "right"
            print(f"우회전: {steering_angle}도")
        elif key == 'q':
            print("종료")
            break
        else:
            motor_stop()
            throttle = 0
            direction = "stop"
            print("정지")

        # 프레임에 방향 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Direction: {direction}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 이미지 저장 (파일명에 방향 포함)
        filename = f"data/frame_{frame_count:05d}_{direction}.jpg"
        cv2.imwrite(filename, frame)
        frame_count += 1

finally:
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam.stop()
    cv2.destroyAllWindows()
