import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
import pigpio
from PIL import Image
from picamera2 import Picamera2

# === 설정 ===
SERVO_PIN = 18
IN1 = 12
IN2 = 13
PWM_SPEED = 120  # 모터 속도
MODEL_PATH = "model_regression.pth"
IMAGE_SIZE = (160, 120)

# === Raspberry Pi GPIO 초기화 ===
pi = pigpio.pi()
pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)
pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

def set_servo_angle(angle):
    angle = max(0, min(180, angle))
    pulsewidth = 500 + (angle / 180.0) * 2000
    pi.set_servo_pulsewidth(SERVO_PIN, pulsewidth)

def motor_forward(speed=PWM_SPEED):
    pi.set_PWM_dutycycle(IN1, speed)
    pi.set_PWM_dutycycle(IN2, 0)

def motor_stop():
    pi.set_PWM_dutycycle(IN1, 0)
    pi.set_PWM_dutycycle(IN2, 0)

# === 모델 정의 ===
class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 8, 100), nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# === 모델 불러오기 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === 이미지 전처리 ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === 카메라 초기화 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

# === 주행 루프 ===
print("start (Ctrl+C로 종료)")

try:
    while True:
        frame = picam2.capture_array()
        pil_image = Image.fromarray(frame)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            norm_angle = output.item()
            angle = int(norm_angle * 180)
            angle = max(0, min(180, angle))

        set_servo_angle(angle)
        motor_forward()

        # 시각화 (선택사항)
        cv2.putText(frame, f"Steering: {angle}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Live Drive", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print(" exit (Ctrl+C)")

finally:
    print("exit")
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("✅ 안전 종료 완료")
