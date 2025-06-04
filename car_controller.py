import RPi.GPIO as GPIO
import time
from pid_simple import PID  # 외부 PID 라이브러리 사용

class CarController:
    def __init__(self, frame_width=320):
        # GPIO 핀 설정
        self.STEER_IN1 = 5
        self.STEER_IN2 = 6
        self.DRIVE_IN1 = 17
        self.DRIVE_IN2 = 18
        self.DRIVE_EN = 12

        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.STEER_IN1, self.STEER_IN2, self.DRIVE_IN1, self.DRIVE_IN2], GPIO.OUT)
        GPIO.setup(self.DRIVE_EN, GPIO.OUT)

        self.drive_pwm = GPIO.PWM(self.DRIVE_EN, 1000)
        self.drive_pwm.start(0)

        # PID 설정 (카메라 프레임 중심 기준)
        self.frame_center = frame_width // 2
        self.pid = PID(P=0.5, I=0.01, D=0.1, setpoint=self.frame_center)

    def set_steering(self, correction):
        if correction < -20:
            GPIO.output(self.STEER_IN1, GPIO.LOW)
            GPIO.output(self.STEER_IN2, GPIO.HIGH)  # 왼쪽
        elif correction > 20:
            GPIO.output(self.STEER_IN1, GPIO.HIGH)
            GPIO.output(self.STEER_IN2, GPIO.LOW)   # 오른쪽
        else:
            GPIO.output(self.STEER_IN1, GPIO.LOW)
            GPIO.output(self.STEER_IN2, GPIO.LOW)   # 직진

    def set_drive(self, speed=60):
        GPIO.output(self.DRIVE_IN1, GPIO.HIGH)
        GPIO.output(self.DRIVE_IN2, GPIO.LOW)
        self.drive_pwm.ChangeDutyCycle(speed)

    def stop_drive(self):
        GPIO.output(self.DRIVE_IN1, GPIO.LOW)
        GPIO.output(self.DRIVE_IN2, GPIO.LOW)
        self.drive_pwm.ChangeDutyCycle(0)

    def update(self, lane_center):
        # lane_center: 차선 중심값 입력
        correction = self.pid.update(lane_center)
        self.set_steering(correction)
        self.set_drive()

    def cleanup(self):
        self.stop_drive()
        GPIO.cleanup()
