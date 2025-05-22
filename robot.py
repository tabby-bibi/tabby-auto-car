# RC카 모드로 직접 조향이 가능 하게 해주는 코드

import RPi.GPIO as GPIO
import pygame
import sys
import time
from pygame.locals import *

# 핀 번호
IN1 = 19  # 후륜 구동
IN2 = 13
ENA = 26

IN3 = 6   # 조향 모터 
IN4 = 5   # 앞바퀴
ENB = 21

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

pins = [IN1, IN2, ENA, IN3, IN4, ENB]
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)

pwm_drive = GPIO.PWM(ENA, 100)
pwm_steer = GPIO.PWM(ENB, 100)
pwm_drive.start(0)
pwm_steer.start(0)

# 구동 함수
def forward(speed=70):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm_drive.ChangeDutyCycle(speed)

# 후진 코드
def reverse(speed=70): 
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm_drive.ChangeDutyCycle(speed)

# 좌회전 코드
def steer_left(duration=0.4, speed=70): 
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_steer.ChangeDutyCycle(speed)
    time.sleep(duration)
    pwm_steer.ChangeDutyCycle(0)  # 정지

# 우회전 코드
def steer_right(duration=0.4, speed=70):
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_steer.ChangeDutyCycle(speed)
    time.sleep(duration)
    pwm_steer.ChangeDutyCycle(0)  # 정지

# 정지 코드
def stop_all():
    pwm_drive.ChangeDutyCycle(0)
    pwm_steer.ChangeDutyCycle(0)
    for pin in [IN1, IN2, IN3, IN4]:
        GPIO.output(pin, GPIO.LOW)

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("DC 조향 차량")
pygame.mouse.set_visible(0)

# 실행 코드 루프문
try:
    print("↑: 전진 / ↓: 후진 / ←: 좌조향 / →: 우조향 / Space: 정지")
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                GPIO.cleanup()
                sys.exit()

            if event.type == KEYDOWN:
                
                if event.key == K_UP:
                    print("전진")
                    forward()
                if event.key == K_DOWN:
                    print("후진")
                    reverse()
                if event.key == K_LEFT:
                    print("좌회전")
                    steer_left()
                if event.key == K_RIGHT:
                    print("우회전")
                    steer_right()
                if event.key == K_SPACE:
                    print("정지")
                    stop_all()

except KeyboardInterrupt:
    print("\n종료됨")


finally:
    stop_all()
    pwm_drive.stop()
    pwm_steer.stop()
    GPIO.cleanup()
    pygame.quit()
