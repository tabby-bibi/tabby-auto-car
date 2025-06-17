# 자율주행 딥런닝 모델 실제 주행 코드

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from picamera2 import Picamera2
import pigpio
import time

# --- 하드웨어 설정 ---
SERVO_PIN = 18
IN1 = 12
IN2 = 13
motor_speed = 40

# --- 모델 정의 (SimpleCNN 구조 동일하게) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 모델 불러오기 ---
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# --- 전처리 정의 ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- GPIO 초기화 ---
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

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

# --- 카메라 초기화 ---
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

print("🚗 실시간 자율주행 시작 (Ctrl+C로 종료)")

try:
    while True:
        frame = picam2.capture_array()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(1).item()

        # 예측값에 따른 조향
        if prediction == 0:  # left
            set_servo_angle(50)
            print("← 좌회전")
        elif prediction == 1:  # center
            set_servo_angle(90)
            print("↑ 직진")
        elif prediction == 2:  # right
            set_servo_angle(130)
            print("→ 우회전")

        motor_forward()
        time.sleep(0.1)  # 주행 간격
        motor_stop()

except KeyboardInterrupt:
    print("🛑 자율주행 종료")

finally:
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
