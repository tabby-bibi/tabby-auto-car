## Raspberry Pi-Based Autonomous RC Car using CNN Regression

## Project Overview

This project implements an autonomous RC car using a Raspberry Pi 4, Pi Camera, servo motor, and DC motor.
A CNN regression model predicts the steering angle from camera input, enabling real-time autonomous driving.

The project includes:

-Data collection system

-CNN model training pipeline

-Real-time autonomous driving control system

## Project Specifications

| Item            | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| Platform        | Raspberry Pi 4 (64-bit)                                             |
| Camera          | Pi Camera 2                                                         |
| Motor Driver    | L298N                                                               |
| Drive Motor     | 1 DC motor                                                          |
| Steering Motor  | 1 SG90 Servo motor                                                  |
| Training Method | CNN-based regression model (PyTorch)                                |
| Data Collection | Manual driving with synchronized image and steering angle recording |
| Model Output    | Steering angle normalized between -1.0 and 1.0                      |





## 하드웨어 구성

| 부품 | 역할 |
|------|------|
| Raspberry Pi 4 | 메인 컨트롤러 |
| Pi Camera 2 | 전방 영상 입력 |
| L298N | 모터 드라이버 |
| DC 모터 | 차량 구동 |
| SG90 서보모터 | 조향 제어 |
| 9v li ion battery 배터리 × 2 | 전원 공급 |
| 휴대용 보조배터리 | 전원 공급 |

![Image](https://github.com/user-attachments/assets/fcb2a593-a4c5-47a7-858f-cb33cb5ece63)
![Image](https://github.com/user-attachments/assets/2389b943-367e-4547-b5c7-17945c0cd248)
![Image](https://github.com/user-attachments/assets/dd00ad1c-0ec1-4d4b-8b88-abcdca8bf464)










