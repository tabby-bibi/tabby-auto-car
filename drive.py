import RPi.GPIO as GPIO
import time
from pid_simple import PID  # pip로 설치한 PID 라이브러리

# --- GPIO 핀 설정 ---
STEER_IN1 = 5
STEER_IN2 = 6
DRIVE_IN1 = 17
DRIVE_IN2 = 18
DRIVE_EN = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup([STEER_IN1, STEER_IN2, DRIVE_IN1, DRIVE_IN2], GPIO.OUT)
GPIO.setup(DRIVE_EN, GPIO.OUT)
drive_pwm = GPIO.PWM(DRIVE_EN, 1000)
drive_pwm.start(0)

# --- PID 컨트롤러 생성 ---
# 목표값: 중앙값 (카메라 프레임 320이면 중앙은 160)
pid = PID(P=0.5, I=0.01, D=0.1, setpoint=160)

# --- 조향 제어 함수 ---
def set_steering(angle):
    if angle < -20:
        GPIO.output(STEER_IN1, GPIO.LOW)
        GPIO.output(STEER_IN2, GPIO.HIGH)  # 왼쪽 조향
    elif angle > 20:
        GPIO.output(STEER_IN1, GPIO.HIGH)
        GPIO.output(STEER_IN2, GPIO.LOW)   # 오른쪽 조향
    else:
        GPIO.output(STEER_IN1, GPIO.LOW)
        GPIO.output(STEER_IN2, GPIO.LOW)   # 직진 유지

# --- 구동 제어 함수 ---
def set_drive(speed):
    GPIO.output(DRIVE_IN1, GPIO.HIGH)
    GPIO.output(DRIVE_IN2, GPIO.LOW)
    drive_pwm.ChangeDutyCycle(speed)

def stop_drive():
    GPIO.output(DRIVE_IN1, GPIO.LOW)
    GPIO.output(DRIVE_IN2, GPIO.LOW)
    drive_pwm.ChangeDutyCycle(0)

# --- 차선 중심값 받아오는 함수 (따로 구현한 모듈 사용) ---/
# def detect_lane_center():
    # 이 함수는 OpenCV로 만든 별도 파일에서 불러온다고 가정
    # 예시로 140~180 사이의 값을 리턴한다고 가정 /
    # return 145

# --- 메인 루프 ---
try:
    while True:
        lane_center = detect_lane_center()
        correction = pid.update(lane_center)  # PID 오차 계산

        set_steering(correction)
        set_drive(60)

        time.sleep(0.05)

except KeyboardInterrupt:
    stop_drive()
    GPIO.cleanup()
