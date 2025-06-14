import cv2
import RPi.GPIO as GPIO
import csv
import time
import sys
import termios
import tty
import os

# GPIO 핀 설정
SERVO_PIN = 18
IN1 = 12
IN2 = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# PWM 설정
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 서보: 50Hz
motor_pwm_in1 = GPIO.PWM(IN1, 1000)  # 모터: 1kHz
motor_pwm_in2 = GPIO.PWM(IN2, 1000)

servo_pwm.start(7.5)        # 90도 중립
motor_pwm_in1.start(0)
motor_pwm_in2.start(0)

# 서보 각도 설정 함수 (0~180도)
def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 180.0) * 10  # 2.5 ~ 12.5
    servo_pwm.ChangeDutyCycle(duty)

# 모터 제어 (속도는 0~100%)
def motor_forward(speed=40):
    motor_pwm_in1.ChangeDutyCycle(speed)
    motor_pwm_in2.ChangeDutyCycle(0)

def motor_backward(speed=40):
    motor_pwm_in1.ChangeDutyCycle(0)
    motor_pwm_in2.ChangeDutyCycle(speed)

def motor_stop():
    motor_pwm_in1.ChangeDutyCycle(0)
    motor_pwm_in2.ChangeDutyCycle(0)

# 키 입력 받는 함수 (blocking)
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)

# 저장 디렉토리 생성
os.makedirs("data", exist_ok=True)
csv_file = open("data/drive_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "steering_angle", "throttle"])

# 초기값
frame_count = 0
steering_angle = 90  # 중립
throttle = 0

set_servo_angle(steering_angle)
motor_stop()

print("RC 모드 시작!")
print("W: 전진 | S: 후진 | A: 좌회전 | D: 우회전 | Q: 종료")

try:
    while True:
        # 영상 캡처
        ret, frame = cap.read()
        if not ret:
            print("카메라 캡처 실패")
            break

        # 영상 보여주기
        cv2.imshow("Driving", frame)
        cv2.waitKey(1)

        # 키 입력 (blocking)
        key = getkey().lower()

        if key == 'w':
            throttle = 1
            motor_forward(40)
            print("전진")
        elif key == 's':
            throttle = -1
            motor_backward(40)
            print("후진")
        elif key == 'a':
            steering_angle = max(0, steering_angle - 30)
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
            throttle = 0
            motor_stop()
            print("정지")

        # 이미지 저장
        filename = f"data/frame_{frame_count:05d}.jpg"
        cv2.imwrite(filename, frame)

        # 로그 저장
        csv_writer.writerow([frame_count, steering_angle, throttle])
        frame_count += 1

finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    motor_stop()
    servo_pwm.stop()
    motor_pwm_in1.stop()
    motor_pwm_in2.stop()
    GPIO.cleanup()
