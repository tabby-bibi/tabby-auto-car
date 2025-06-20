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
PWM_SPEED = 110  # 모터 속도
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

# === 모델 정의 및 불러오기 ===
class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 17 * 12, 100), nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === 이미지 전처리 ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === 카메라 초기화 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()

print("▶ 자율주행 시작 (Ctrl+C로 종료)")

THRESHOLD = 1000  # 차선 인식 신뢰도 임계값 (환경에 맞게 조정하세요)

try:
    while True:
        frame = picam2.capture_array()

        # 차선 인식 (ROI 하단 1/4 영역)
        roi = frame[360:480, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        histogram = np.sum(edges, axis=0)
        midpoint = histogram.shape[0] // 2
        leftx = np.argmax(histogram[:midpoint])
        rightx = np.argmax(histogram[midpoint:]) + midpoint
        center_x = (leftx + rightx) // 2

        lane_signal = histogram[leftx:rightx].sum()
        frame_center = frame.shape[1] // 2
        error = center_x - frame_center

        # 차선 인식 신뢰도 체크
        if lane_signal < THRESHOLD:
            # 차선 인식 실패 -> 정지
            motor_stop()
            cv2.putText(frame, "No Lane Detected - STOP", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3, cv2.LINE_AA)
            print("⚠️ 차선 인식 실패 - 정지")
        else:
            # 차선 인식 성공 -> 딥러닝 모델 예측 이용
            pil_image = Image.fromarray(frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                norm_angle = output.item()
                angle = int(norm_angle * 180)
                angle = max(0, min(180, angle))

            set_servo_angle(angle)
            motor_forward()

            cv2.putText(frame, f"Steering: {angle}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # 시각화 - 차선 중심선, 화면 중심선 표시
        cv2.line(frame, (center_x, 460), (center_x, 480), (255, 0, 0), 3)   # 파란선 - 차선 중심
        cv2.line(frame, (frame_center, 460), (frame_center, 480), (0, 0, 255), 3)  # 빨간선 - 화면 중심

        cv2.imshow("Live Drive", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("▶ 강제 종료됨 (Ctrl+C)")

finally:
    print("▶ 정리 중...")
    motor_stop()
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()
    print("✅ 안전 종료 완료")
