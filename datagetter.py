import RPi.GPIO as GPIO
import time
import cv2
import csv
import os
import sys
import termios
import tty
from picamera2 import Picamera2
import numpy as np

# ==========================
# 핀 설정
# ==========================
SERVO_PIN = 18
IN1 = 17
IN2 = 27

# ==========================
# GPIO 초기화
# ==========================
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# 서보 PWM (50Hz)
servo_pwm = GPIO.PWM(SERVO_PIN, 50)
servo_pwm.start(0)

# ==========================
# 서보모터 제어 함수
# ==========================
def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 180.0) * 10  # 0도: 2.5%, 180도: 12.5%
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.05)  # 서보모터 반응 시간

# ==========================
# DC 모터 제어 함수
# ==========================
def motor_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

def motor_backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)

# ==========================
# 키 입력 처리 함수 (blocking)
# ==========================
def getkey():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# ==========================
# PiCamera2 초기화
# ==========================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (320, 240)}))
picam2.start()

# ==========================
# 데이터 저장 준비
# ==========================
os.makedirs('data', exist_ok=True)
csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

frame_count = 0
steering_angle = 90  # 중립
throttle = 0         # -1: 후진, 0: 정지, 1: 전진

print("▶ 키보드 조작 안내: W(전진) S(후진) A(좌) D(우) Q(종료)")
print("▶ 이미지 저장 및 CSV 라벨 작성 진행 중...")

try:
    while True:
        frame = picam2.capture_array()
        key = getkey().lower()

        # 조작 키 입력 처리
        if key == 'w':
            throttle = 1
            motor_forward()
        elif key == 's':
            throttle = -1
            motor_backward()
        elif key == 'a':
            steering_angle = max(40, steering_angle - 30)
            set_servo_angle(steering_angle)
        elif key == 'd':
            steering_angle = min(140, steering_angle + 30)
            set_servo_angle(steering_angle)
        elif key == 'q':
            print("▶ 종료")
            break
        else:
            throttle = 0
            motor_stop()

        # 화면에 조향 방향 표시
        direction_text = {
            -1: "후진",
            0: "정지",
            1: "전진"
        }[throttle]
        label = f"{direction_text} | 조향: {steering_angle}°"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 이미지 저장
        filename = f'data/frame_{frame_count:05d}.jpg'
        cv2.imwrite(filename, frame)

        # CSV 저장
        csv_writer.writerow([frame_count, steering_angle, throttle])

        # 화면 출력
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        frame_count += 1

finally:
    motor_stop()
    servo_pwm.stop()
    GPIO.cleanup()
    csv_file.close()
    cv2.destroyAllWindows()
    picam2.stop()
