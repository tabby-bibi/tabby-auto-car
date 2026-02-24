Raspberry Pi-Based Autonomous RC Car using CNN Regression
Project Overview

This project implements an autonomous RC car using a Raspberry Pi 4, Pi Camera, servo motor, and DC motor.
A CNN regression model predicts the steering angle from camera input, enabling real-time autonomous driving.

The project includes:

Data collection system

CNN model training pipeline

Real-time autonomous driving control system

Project Specifications
Item	Description
Platform	Raspberry Pi 4 (64-bit)
Camera	Pi Camera 2
Motor Driver	L298N
Drive Motor	1 DC motor
Steering Motor	1 SG90 Servo motor
Training Method	CNN-based regression model (PyTorch)
Data Collection	Manual driving with synchronized image and steering angle recording
Model Output	Steering angle normalized between -1.0 and 1.0




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



## 데이터 수집 과정

`frame_save.py`를 실행하여 조종기로 차량을 조작하면  
카메라 영상과 함께 조향각(servo angle)을 CSV 파일로 기록합니다.

frame, steering_angle
image_001.jpg, 10
image_002.jpg, -15
...

데이터는 약 3000장 이상 수집되어 모델 학습에 사용되었습니다.




## CNN 회귀 모델 구조 ##

https://github.com/tabby-bibi/tabby-auto-car/blob/main/CNN%ED%9A%8C%EA%B7%80%EB%AA%A8%EB%8D%B8.ipynb



## 학습 및 모델 저장 ##

CNN회귀모델.ipynb 실행 시 다음 순서로 진행됩니다.

이미지 로드 - 전처리 - CNN 학습 - 학습 완료 후 가중치 저장


## 실시간 자율주행 ##

Raspberry Pi에서 카메라 영상 실시간 입력

전처리 후 모델에 입력

모델 예측값 → 조향각(servo)으로 변환

DC 모터 일정 속도 유지로 전진

| **항목**           |         **결과**        |
| :--------------- | :-------------------: |
| **주행 성공률**    | 약 **90%** (테스트 트랙 기준) |
| **평균 조향 오차**  |        ±**3°**        |
|  **평균 속도**     |     약 **0.4 m/s**     |
| **학습 이미지 수** |      약 **3,000장**     |
| **모델 파라미터 수** |       약 **500K**      |


# 참고 사항
데이터셋은 라즈베리파이에서 직접 수집 (수동 조종 기반)

모델 학습은 Google Colab + PyTorch로 수행

학습 후 .pth 모델 파일을 Raspberry Pi로 전송하여 주행 테스트

회귀모델을 사용함으로써 분류모델 대비 더 자연스러운 steering 제어 가능

## 향후 개선 방향 ##

장애물 인식과 장애물 회피 기능 추가

속도 제어 알고리즘 개선 (강화학습)

실시간 모니터링 기능 추가

실시간 카메라 방향 전환 기능 추가 

-------------------------------------










## 참고자료

http://lhdangerous.godohosting.com/wiki/index.php/Raspberry_pi_%ec%97%90%ec%84%9c_python%ec%9c%bc%eb%a1%9c_GPIO_%ec%82%ac%ec%9a%a9%ed%95%98%ea%b8%b0

https://github.com/JD-edu/deepThinkCar_mini






