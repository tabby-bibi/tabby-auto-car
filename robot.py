# RC카 모드로 직접 조향이 가능 하게 해주는 코드

import pigpio
import time
import sys
import termios
import tty

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

# 키보드 입력 받는 함수
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# 초기 상태
steering_angle = 90  # 중립

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
            steering_angle = min(140, steering_angle + 30)
            set_servo_angle(steering_angle)
            print(f"우회전: {steering_angle}도")
        elif key == 'q':
            print("종료")
            break
        else:
            motor_stop()
            print("정지")

finally:
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
