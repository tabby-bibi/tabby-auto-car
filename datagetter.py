import pigpio
import time
import sys
import termios
import tty
import os
import cv2
import csv
import threading
from picamera2 import Picamera2

# === 설정 ===
SERVO_PIN = 18
IN1 = 12
IN2 = 13
SAVE_DIR = "data"
FRAME_SAVE = True

motor_speed = 40  # PWM 속도 (0~255)

# === 초기화 ===
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward(speed=motor_speed):
    speed = max(0, min(255, speed))
    pi.set_PWM_dutycycle(IN1, speed)
    pi.set_PWM_dutycycle(IN2, 0)

def motor_backward(speed=motor_speed):
    speed = max(0, min(255, speed))
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, speed)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# === 카메라 초기화 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

# === 글로벌 변수 ===
latest_frame = None
running = True

def camera_thread():
    global latest_frame, running
    while running:
        frame = picam2.capture_array()
        latest_frame = frame
        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# === 영상 스레드 시작 ===
threading.Thread(target=camera_thread, daemon=True).start()

# === 저장 폴더 및 CSV 초기화 ===
if FRAME_SAVE:
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_file = open(os.path.join(SAVE_DIR, "drive_log.csv"), mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "steering_angle", "label"])
    frame_count = 0

# 초기 상태
steering_angle = 90
set_servo_angle(steering_angle)
motor_stop()

print("조작 키 안내:")
print(" W : 전진 | S : 후진")
print(" A : 좌회전 | D : 우회전")
print(" Q : 종료")

try:
    while running:
        key = getkey().lower()

        if key == 'w':
            motor_forward(motor_speed)
            print(f"전진 (속도: {motor_speed})")
        elif key == 's':
            motor_backward(motor_speed)
            print(f"후진 (속도: {motor_speed})")
        elif key == 'a':
            steering_angle = max(40, steering_angle - 30)
            set_servo_angle(steering_angle)
            print(f"좌회전: {steering_angle}도")
        elif key == 'd':
            steering_angle = min(180, steering_angle + 30)
            set_servo_angle(steering_angle)
            print(f"우회전: {steering_angle}도")
        elif key == 'q':
            print("종료")
            running = False
            break
        else:
            motor_stop()
            print("정지")

        # 이미지 저장 및 라벨링
        if FRAME_SAVE and latest_frame is not None and key in ['w', 's']:
            filename = f"frame_{frame_count:05d}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, latest_frame)

            if steering_angle < 70:
                label = "left"
            elif steering_angle > 110:
                label = "right"
            else:
                label = "center"

            csv_writer.writerow([filename, steering_angle, label])
            frame_count += 1

except KeyboardInterrupt:
    print("Ctrl+C 강제 종료 감지됨")

finally:
    running = False
    time.sleep(0.5)  # 스레드 종료 대기

    print("=== 종료 정리 중 ===")
    if FRAME_SAVE:
        csv_file.close()

    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)

    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("모든 자원 해제 완료. 안전 종료되었습니다.")
