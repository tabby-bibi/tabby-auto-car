# car_controller.py
import RPi.GPIO as GPIO
import time

class CarController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.IN1 = 22
        self.IN2 = 23
        self.ENA = 27

        self.IN3 = 16
        self.IN4 = 19
        self.ENB = 21

        GPIO.setup(self.IN1, GPIO.OUT)
        GPIO.setup(self.IN2, GPIO.OUT)
        GPIO.setup(self.ENA, GPIO.OUT)

        GPIO.setup(self.IN3, GPIO.OUT)
        GPIO.setup(self.IN4, GPIO.OUT)
        GPIO.setup(self.ENB, GPIO.OUT)

        self.drive_pwm = GPIO.PWM(self.ENA, 100)
        self.steer_pwm = GPIO.PWM(self.ENB, 100)

        self.drive_pwm.start(0)
        self.steer_pwm.start(0)

        self.set_motor_forward()

    def set_motor_forward(self):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        self.drive_pwm.ChangeDutyCycle(40)

    def stop_drive(self):
        self.drive_pwm.ChangeDutyCycle(0)
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)

    def stop_steer(self):
        self.steer_pwm.ChangeDutyCycle(0)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)

    def update(self, direction, speed=87, duration=0.5):
        if direction == "left":
            GPIO.output(self.IN3, GPIO.HIGH)
            GPIO.output(self.IN4, GPIO.LOW)
        elif direction == "right":
            GPIO.output(self.IN3, GPIO.LOW)
            GPIO.output(self.IN4, GPIO.HIGH)
        else:
            self.stop_steer()
            return

        self.steer_pwm.ChangeDutyCycle(speed)
        time.sleep(duration)
        self.steer_pwm.ChangeDutyCycle(0)
        self.stop_steer()
