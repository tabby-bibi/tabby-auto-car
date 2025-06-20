import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import threading
import time
from picamera2 import Picamera2
import pigpio

# --- 하드웨어 핀 설정 ---
SERVO_PIN = 18
IN1 = 12
IN2 = 13
motor_speed = 120

# --- 모델 정의 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)  # left, center, right, stop

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 모델 로딩 ---
model = SimpleCNN()
model.load_state_dict(torch.load("pm.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- pigpio 초기화 ---
pi = pigpio.pi()
if not pi.connected:
    raise IOError("pigpio 데몬 연결 실패")
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)
pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward(speed=motor_speed):
    pi.set_PWM_dutycycle(IN1, speed)
    pi.set_PWM_dutycycle(IN2, 0)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

def smooth_angle(prev, new, alpha=0.7):
    return int(alpha * prev + (1 - alpha) * new)

# --- 카메라 초기화 ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'BGR888', "size": (680, 240)})
picam2.configure(config)
picam2.start()
time.sleep(2)

# --- 영상 저장 초기화 ---
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('drive_output.avi', fourcc, 20.0, (680, 240))

# --- 공유 변수 ---
latest_frame = None
frame_lock = threading.Lock()
driving = True

# --- 주행 쓰레드 ---
def driving_thread():
    global latest_frame, driving
    prev_angle = 90

    while driving:
        with frame_lock:
            if latest_frame is None:
                continue
            image = Image.fromarray(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            prediction = torch.argmax(probs).item()

        if prediction == 3:  # stop
            motor_stop()
            print("■ 정지")
            time.sleep(0.1)
            continue

        if prediction == 0:  # 좌회전
            target_angle = int(60 - 30 * probs[0].item())  # 60~30도
            print(f"← 좌회전 ({target_angle}도)")
        elif prediction == 1:  # 직진
            target_angle = 90
            print("↑ 직진")
        elif prediction == 2:  # 우회전
            target_angle = int(120 + 30 * probs[2].item())  # 120~150도
            print(f"→ 우회전 ({target_angle}도)")

        angle = smooth_angle(prev_angle, target_angle)
        set_servo_angle(angle)
        motor_forward()
        prev_angle = angle

        time.sleep(0.1)

# --- 주행 쓰레드 시작 ---
t = threading.Thread(target=driving_thread)
t.start()

# --- 영상 띄우기 & 저장 루프 ---
try:
    print("🚗 자율주행 시작 (q 키 종료)")
    while True:
        frame = picam2.capture_array()

        with frame_lock:
            latest_frame = frame.copy()

        cv2.imshow("📷 실시간 주행 영상", frame)
        video_out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🚨 강제 종료")

finally:
    driving = False
    t.join()

    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
    video_out.release()
    cv2.destroyAllWindows()
    print("🛑 주행 종료 및 영상 저장 완료")
