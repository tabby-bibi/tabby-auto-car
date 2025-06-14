import pigpio
import time
import sys
import termios
import tty
import cv2
from picamera2 import Picamera2

# 핀 번호 설정
SERVO_PIN = 18   # 서보모터 (조향)
IN1 = 12         # DC 모터 제어
IN2 = 13

# pigpio 초기화
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)

# 서보 각도 설정 함수 (0~180도)
def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000  # 500~2500us
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

# 키보드 입력 받는 함수 (블로킹)
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# 텍스트를 이미지에 넣는 함수
def put_direction_text(img, direction):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Direction: {direction}", (10, 30), font, 1, (0, 0, 255), 2)

# 초기 상태
steering_angle = 90  # 중립

import os
os.makedirs('data', exist_ok=True)

# PiCamera2 초기화
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": (320, 240)}))
picam.start()
time.sleep(1)  # 카메라 워밍업

frame_count = 0

print("조작 키 안내:")
print(" W : 전진")
print(" S : 후진")
print(" A : 좌회전")
print(" D : 우회전")
print(" Q : 종료")

set_servo_angle(steering_angle)
motor_stop()

try:
    while True:
        frame = picam.capture_array()

        key = getkey().lower()

        throttle = 0
        direction = "straight"

        if key == 'w':
            throttle = 40
            motor_forward()
            direction = "straight"
        elif key == 's':
            throttle = -40
            motor_backward()
            direction = "straight"
        elif key == 'a':
            steering_angle = max(40, steering_angle - 20)
            set_servo_angle(steering_angle)
            direction = "left"
        elif key == 'd':
            steering_angle = min(140, steering_angle + 20)
            set_servo_angle(steering_angle)
            direction = "right"
        elif key == 'q':
            print("종료")
            break
        else:
            motor_stop()
            direction = "stop"

        put_direction_text(frame, direction)

        # 이미지 저장
        filename = f"data/frame_{frame_count:05d}_{direction}.jpg"
        cv2.imwrite(filename, frame)
        frame_count += 1

        # 화면 표시
        cv2.imshow("RC Car Driving", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam.stop()
    cv2.destroyAllWindows()
