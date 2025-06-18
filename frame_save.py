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
# IN1 = 12  # 모터 제어 핀, 모터 제어 안함으로 주석 처리
# IN2 = 13
SAVE_DIR = "data"
FRAME_SAVE = True
motor_speed = 110  # PWM 속도 (0~255)

# === 초기화 ===
pi = pigpio.pi()
# pi.set_mode(IN1, pigpio.OUTPUT)
# pi.set_mode(IN2, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

# 모터 제어 함수들 주석 처리 (사용 안 함)
# def motor_forward(speed=motor_speed):
#     pi.set_PWM_dutycycle(IN1, speed)
#     pi.set_PWM_dutycycle(IN2, 0)
#
# def motor_backward(speed=motor_speed):
#     pi.set_PWM_dutycycle(IN1, 0)
#     pi.set_PWM_dutycycle(IN2, speed)
#
# def motor_stop():
#     pi.set_PWM_dutycycle(IN1, 0)
#     pi.set_PWM_dutycycle(IN2, 0)

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

threading.Thread(target=camera_thread, daemon=True).start()

# === 저장 폴더 및 CSV 초기화 ===
if FRAME_SAVE:
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_file = open(os.path.join(SAVE_DIR, "drive_log.csv"), mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "steering_angle", "label"])
    frame_count = 0

steering_angle = 90
set_servo_angle(steering_angle)
# motor_stop()  # 모터 제어 안 함

print("조작 키 안내:")
print(" W : 전진 | S : 후진")
print(" A : 좌회전 | D : 우회전")
print(" 스페이스바 : 정지 상태 저장")
print(" Q : 종료")

saving_frames = False

try:
    while running:
        key = getkey().lower()

        saving_frames = False
        label = "stop"  # 기본값 stop으로 시작

        if key == 'w':
            # motor_forward(motor_speed)  # 모터 제어 안함
            print(f"전진 (속도: {motor_speed})")
            saving_frames = True
        elif key == 's':
            # motor_backward(motor_speed)  # 모터 제어 안함
            print(f"후진 (속도: {motor_speed})")
            saving_frames = True
        elif key == 'a':
            steering_angle = max(0, steering_angle - 20)
            set_servo_angle(steering_angle)
            print(f"좌회전: {steering_angle}도")
            saving_frames = True
        elif key == 'd':
            steering_angle = min(180, steering_angle + 20)
            set_servo_angle(steering_angle)
            print(f"우회전: {steering_angle}도")
            saving_frames = True
        elif key == ' ':
            # motor_stop()  # 모터 제어 안함
            print("정지 상태 저장")
            saving_frames = True
            label = "stop"
        elif key == 'q':
            print("종료")
            running = False
            break
        else:
            # motor_stop()  # 모터 제어 안함
            print("정지")
            saving_frames = False
            continue

        if label != "stop":
            if steering_angle < 70:
                label = "left"
            elif steering_angle > 110:
                label = "right"
            else:
                label = "center"

        # 라벨 텍스트를 프레임에 그려서 저장
        if FRAME_SAVE and latest_frame is not None and saving_frames:
            frame_with_label = latest_frame.copy()
            text = label.upper()
            org = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0)
            thickness = 2
            cv2.putText(frame_with_label, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

            filename = f"frame_{frame_count:05d}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame_with_label)
            csv_writer.writerow([filename, steering_angle, label])
            print(f"저장: {filename} - {label}")
            frame_count += 1

except KeyboardInterrupt:
    print("Ctrl+C 강제 종료 감지됨")

finally:
    running = False
    time.sleep(0.5)

    print("=== 종료 정리 중 ===")
    if FRAME_SAVE:
        csv_file.close()

    # motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("모든 자원 해제 완료. 안전 종료되었습니다.")
