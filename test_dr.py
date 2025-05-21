import RPi.GPIO as GPIO
import time

# BCM 모드 설정
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 핀 번호 설정
ENA = 26  # 앞 모터 PWM
ENB = 21  # 뒤 모터 PWM (GPIO0 대신 GPIO21 사용 권장)

IN1 = 19  # 앞 모터 방향
IN2 = 13
IN3 = 6   # 뒤 모터 방향
IN4 = 5

# GPIO 출력 모드 설정
motor_pins = [ENA, ENB, IN1, IN2, IN3, IN4]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# PWM 핀 초기화
pwm1 = GPIO.PWM(ENA, 1000)  # 앞 모터 PWM, 1kHz
pwm2 = GPIO.PWM(ENB, 1000)  # 뒤 모터 PWM

# PWM 시작 (속도 80%)
pwm1.start(80)
pwm2.start(80)

# 앞/뒤 모터 정방향 회전 설정
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)

print("모터 무한 회전 중... (Ctrl+C로 종료)")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("종료 중...")
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()
