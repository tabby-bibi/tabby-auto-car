import cv2
import time
import csv
import pigpio
import sys
import tty
import termios
from picamera2 import Picamera2

# pigpio 데몬에 연결
pi = pigpio.pi()
if not pi.connected:
    print("pigpiod가 실행 중인지 확인하세요.")
    sys.exit()

# 핀 설정
SERVO_PIN = 18
IN1 = 17
IN2 = 27

pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)

# 서보모터 각도 설정
def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = int(500 + (angle / 180.0) * 2000)
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

# 모터 속도 설정 (0~255)
def motor_forward(speed=255):
    pi.set_PWM_dutycycle(IN1, speed)
    pi.write(IN2, 0)

def motor_backward(speed=255):
    pi.set_PWM_dutycycle(IN2, speed)
    pi.write(IN1, 0)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

# 키 입력 함수 (blocking 방식)
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return key

# 카메라 초기화
cam = Picamera2()
cam.preview_configuration.main.size = (320, 240)
cam.preview_configuration.main.format = "RGB888"
cam.configure("preview")
cam.start()

# 데이터 저장 준비
import os
os.makedirs("data", exist_ok=True)
csv_file = open('data/drive_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'steering_angle', 'throttle'])

frame_count = 0
steering_angle = 90
throttle = 0  # 1: 전진, -1: 후진, 0: 정지
speed = 40   # 모터 속도 (0~255)

print("조작 키: W(전진), S(후진), A(좌), D(우), Q(종료)")

try:
    while True:
        # 영상 촬영
        frame = cam.capture_array()

        # 키 입력
        key = getkey().lower()

        if key == 'w':
            throttle = 1
            motor_forward(speed)
        elif key == 's':
            throttle = -1
            motor_backward(speed)
        elif key == 'a':
            steering_angle = max(0, steering_angle - 5)
            set_servo_angle(steering_angle)
        elif key == 'd':
            steering_angle = min(180, steering_angle + 5)
            set_servo_angle(steering_angle)
        elif key == 'q':
            print("종료")
            break
        else:
            throttle = 0
            motor_stop()

        # 이미지 저장
        filename = f"data/frame_{frame_count:05d}.jpg"
        cv2.imwrite(filename, frame)

        # CSV 저장
        csv_writer.writerow([filename, steering_angle, throttle])
        frame_count += 1

        # 실시간 영상 출력
        display_frame = frame.copy()
        direction = "STOP" if throttle == 0 else "FORWARD" if throttle == 1 else "BACKWARD"
        cv2.putText(display_frame, f"{direction}, Angle: {steering_angle}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Driving Preview", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    csv_file.close()
    cam.stop()
    cv2.destroyAllWindows()
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
