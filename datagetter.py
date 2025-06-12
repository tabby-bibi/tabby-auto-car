import cv2
import numpy as np
import pigpio
import csv
import time
from threading import Thread

# 서보 & 모터 제어 핀 정의
SERVO_PIN = 18
IN1 = 12
IN2 = 13

# pigpio 초기화
pi = pigpio.pi()

# 서보 중립값 (PWM duty cycle)
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

# 키 입력으로 조향과 속도 제어
import sys
import termios
import tty

def getkey():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# 영상 캡처 초기화
cap = cv2.VideoCapture(0)

# 저장 준비
csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

frame_count = 0
steering_angle = 90  # 중립
throttle = 0  # 0: 정지, 1: 전진, -1: 후진

try:
    print("W: 전진, S: 후진, A: 좌회전, D: 우회전, Q: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = None
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = getkey().lower()

        # 키보드 입력에 따른 조향, 속도 변경
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

        # 영상 저장 (파일명 예: data/frame_00001.jpg)
        cv2.imwrite(f'data/frame_{frame_count:05d}.jpg', frame)

        # CSV 기록
        csv_writer.writerow([frame_count, steering_angle, throttle])

        frame_count += 1

        # 영상 화면 표시 (원하면)
        cv2.imshow('Driving', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
