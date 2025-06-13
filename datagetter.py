import os
import cv2
import numpy as np
import csv
import time
import pigpio
from picamera2 import Picamera2

# 핀 정의
SERVO_PIN = 18
IN1 = 12
IN2 = 13

# pigpio 초기화
pi = pigpio.pi()

# 서보 제어 함수
def set_servo_angle(angle):
    pulsewidth = int(500 + (angle / 180.0) * 2000)
    pulsewidth = max(500, min(2500, pulsewidth))
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

# 모터 제어 함수
def motor_forward():
    pi.write(IN1, 1)
    pi.write(IN2, 0)

def motor_backward():
    pi.write(IN1, 0)
    pi.write(IN2, 1)

def motor_stop():
    pi.write(IN1, 0)
    pi.write(IN2, 0)

# 카메라 설정
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()
time.sleep(2)

# 디렉토리 및 CSV
os.makedirs("data", exist_ok=True)
csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

# 초기값
frame_count = 0
steering_angle = 90  # 직진
throttle = 0         # 1: 전진, -1: 후진, 0: 정지

print("↑: 전진, ↓: 후진, ←: 좌회전, →: 우회전, space: 정지, q: 종료")

try:
    while True:
        frame = picam2.capture_array()

        key = cv2.waitKey(1) & 0xFF

        # 방향키 처리
        if key == ord('q'):
            break
        elif key == 82:  # ↑
            throttle = 1
        elif key == 84:  # ↓
            throttle = -1
        elif key == 32:  # spacebar
            throttle = 0
        elif key == 81:  # ←
            steering_angle = max(0, steering_angle - 5)
            set_servo_angle(steering_angle)
        elif key == 83:  # →
            steering_angle = min(180, steering_angle + 5)
            set_servo_angle(steering_angle)

        # 모터 동작 지속
        if throttle == 1:
            motor_forward()
        elif throttle == -1:
            motor_backward()
        else:
            motor_stop()

        # 프레임 저장
        filename = f'data/frame_{frame_count:05d}.jpg'
        cv2.imwrite(filename, frame)

        # CSV 기록
        csv_writer.writerow([frame_count, steering_angle, throttle])
        frame_count += 1

        # 화면 출력
        cv2.imshow("Driving", frame)

finally:
    csv_file.close()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    cv2.destroyAllWindows()
