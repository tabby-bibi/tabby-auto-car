import pigpio
import time
import sys
import termios
import tty
import os
import cv2
import csv
import glob
import threading
from picamera2 import Picamera2

# === 설정 ===
SERVO_PIN = 18
IN1 = 12
IN2 = 13
SAVE_DIR = "data"
FRAME_SAVE = True
motor_speed = 110  # PWM 속도 (0~255)

# === 초기화 ===
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward(speed=motor_speed):
    pi.set_PWM_dutycycle(IN1, speed)
    pi.set_PWM_dutycycle(IN2, 0)

def motor_backward(speed=motor_speed):
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
        latest_frame = frame.copy()
        cv2.imshow("Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

threading.Thread(target=camera_thread, daemon=True).start()

# === 저장 폴더 및 CSV 초기화 ===
if FRAME_SAVE:
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 기존 프레임 수만큼 frame_count 이어서 시작
    existing_frames = sorted(glob.glob(os.path.join(SAVE_DIR, "frame_*.jpg")))
    if existing_frames:
        last_frame = existing_frames[-1]
        frame_count = int(os.path.basename(last_frame).split("_")[1].split(".")[0]) + 1
    else:
        frame_count = 0

    # CSV 이어쓰기
    csv_path = os.path.join(SAVE_DIR, "drive_log.csv")
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if not file_exists:
        csv_writer.writerow(["frame", "steering_angle", "label"])

# === 초기 상태 ===
steering_angle = 90
set_servo_angle(steering_angle)
motor_stop()

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
        label = "stop"  # 기본은 정지

        if key == 'w':
            # 실제로는 정지시켜도 라벨링만 한다면 주석 가능
            # motor_forward(motor_speed)
            print("전진")
            label = "center"
            saving_frames = True
        elif key == 's':
            # motor_backward(motor_speed)
            print("후진")
            label = "backward"
            saving_frames = True
        elif key == 'a':
            steering_angle = max(0, steering_angle - 20)
            set_servo_angle(steering_angle)
            print(f"좌회전: {steering_angle}도")
            label = "left"
            saving_frames = True
        elif key == 'd':
            steering_angle = min(180, steering_angle + 20)
            set_servo_angle(steering_angle)
            print(f"우회전: {steering_angle}도")
            label = "right"
            saving_frames = True
        elif key == ' ':
            motor_stop()
            print("정지 상태 저장")
            label = "stop"
            saving_frames = True
        elif key == 'q':
            print("종료")
            running = False
            break
        else:
            motor_stop()
            continue

        # 프레임 저장
        if FRAME_SAVE and latest_frame is not None and saving_frames:
            annotated = latest_frame.copy()
            cv2.putText(annotated, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

            filename = f"frame_{frame_count:05d}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, annotated)
            csv_writer.writerow([filename, steering_angle, label])
            print(f"저장: {filename} - 각도: {steering_angle} - 라벨: {label}")
            frame_count += 1

except KeyboardInterrupt:
    print("강제 종료됨 (Ctrl+C)")

finally:
    running = False
    time.sleep(0.5)

    print("=== 종료 정리 중 ===")
    if FRAME_SAVE:
        csv_file.close()

    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("모든 자원 해제 완료. 안전 종료되었습니다.")
