import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from picamera2 import Picamera2
import pigpio
import time

# --- 핀 번호 설정 ---
SERVO_PIN = 18
IN1 = 12
IN2 = 13
motor_speed = 120  # 모터 속도 (0~255)

# --- SimpleCNN 모델 정의 (출력 클래스 수: 4) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 4)  # ← 4개 클래스 (left, center, right, stop)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64→32
        x = self.pool(F.relu(self.conv2(x)))  # 32→16
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 모델 로드 ---
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# --- 이미지 전처리 정의 ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- pigpio 초기화 ---
pi = pigpio.pi()
if not pi.connected:
    raise IOError("pigpio 데몬에 연결되지 않음")

pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)
pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000  # 500~2500us
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward(speed=motor_speed):
    pi.set_PWM_dutycycle(IN1, speed)
    pi.set_PWM_dutycycle(IN2, 0)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

# --- 카메라 초기화 (PiCamera2) ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": 'BGR888', "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("🚗 실시간 자율주행 시작 (Ctrl+C로 종료)")

try:
    while True:
        frame = picam2.capture_array()  # OpenCV 형식
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(1).item()

        # --- 예측 결과에 따른 제어 ---
        if prediction == 0:  # left
            set_servo_angle(70)
            motor_forward()
            print("← 좌회전")
        elif prediction == 1:  # center
            set_servo_angle(90)
            motor_forward()
            print("↑ 직진")
        elif prediction == 2:  # right
            set_servo_angle(110)
            motor_forward()
            print("→ 우회전")
        elif prediction == 3:  # stop
            motor_stop()
            print(" 정지")

        time.sleep(0.15)
        motor_stop()  # 매 프레임마다 멈춤 (원하면 제거)

except KeyboardInterrupt:
    print("🛑 자율주행 종료")

finally:
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
