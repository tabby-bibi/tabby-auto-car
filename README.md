# Raspberry Pi-Based Autonomous RC Car using CNN Regression

![Image](https://github.com/user-attachments/assets/7adc6bf2-f083-47e5-afa2-2271fa7402fb)

## Project Overview

This project implements an autonomous RC car using a Raspberry Pi 4, Pi Camera, servo motor, and DC motor.
A CNN regression model predicts the steering angle from camera input, enabling real-time autonomous driving.

**Result: ~70% success on the test track, ~0.4 m/s average speed.**

**My part:** the whole pipeline — data collection (`frame_save.py`), CNN training (`CNN_regression_model.ipynb`) and on-car inference (`model_new.py`).

The project includes:

* Data collection system
* CNN model training pipeline
* Real-time autonomous driving control system

---

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
| Model Output    | Steering angle divided by 180 (0.0 = full left, 0.5 = straight, 1.0 = full right) |

---

## Hardware Components

| Component            | Role                      |
| -------------------- | ------------------------- |
| Raspberry Pi 4       | Main controller           |
| Pi Camera 2          | Front-facing video input  |
| L298N                | Motor driver              |
| DC Motor             | Vehicle propulsion        |
| SG90 Servo Motor     | Steering control          |
| 9V Li-ion Battery ×2 | Motor power supply        |
| Portable Power Bank  | Raspberry Pi power supply |

![Image](https://github.com/user-attachments/assets/fcb2a593-a4c5-47a7-858f-cb33cb5ece63)
![Image](https://github.com/user-attachments/assets/2389b943-367e-4547-b5c7-17945c0cd248)
![Image](https://github.com/user-attachments/assets/dd00ad1c-0ec1-4d4b-8b88-abcdca8bf464)

---

## Data Collection Process

The `frame_save.py` script is used to collect training data.

Frames are labelled from the keyboard while the camera looks at the track: `w` = straight, `a` / `d` = steer 20° left / right (the servo moves), `s` = backward, `space` = stop. In this labelling script the drive motor is disabled. Each saved frame (640×480) is written to `drive_log.csv` together with the current servo angle:

```
frame,steering_angle,label
frame_00000.jpg,90,center
frame_00001.jpg,70,left
frame_00002.jpg,110,right
...
```

`steering_angle` is the servo angle in degrees (0–180, 90 = straight). Because the keyboard steps in 20° increments, the labels take a small set of values. A 32-frame sample is included in `databox.zip`.

Approximately **3,000+ images** were collected and used for model training.

---

## CNN Regression Model Architecture

Model notebook: [`CNN_regression_model.ipynb`](./CNN_regression_model.ipynb)

The model predicts a continuous steering angle value rather than using classification.

| Layer | Output shape (C × H × W) | Parameters |
|---|---|---|
| Input (resized) | 3 × 160 × 120 | – |
| Conv 5×5, stride 2, ReLU | 16 × 78 × 58 | 1,216 |
| Conv 5×5, stride 2, ReLU | 32 × 37 × 27 | 12,832 |
| Conv 5×5, stride 2, ReLU | 64 × 17 × 12 | 51,264 |
| Linear 13,056 → 100, ReLU | 100 | 1,305,700 |
| Linear 100 → 1 | 1 | 101 |
| **Total** | | **1,371,113** |

Loss: MSE on `steering_angle / 180`. Optimizer: Adam, lr 1e-3, batch size 32, 20 epochs.

---

## Model Training and Saving

The training process is performed using the `CNN_regression_model.ipynb` notebook.

Training pipeline:

1. Load image dataset
2. Perform image preprocessing
3. Train CNN regression model
4. Save trained model weights (`.pth` file)

Training was performed using:

* Google Colab
* PyTorch

The trained model file is then transferred to the Raspberry Pi for real-time inference.

---

## Real-Time Autonomous Driving System

The autonomous driving process works as follows:

1. Capture real-time video input from Pi Camera
2. Apply preprocessing to the image
3. Feed the image into the trained CNN model
4. Predict steering angle
5. Convert prediction into servo motor control signal
6. Maintain constant forward speed using DC motor

---

## Performance Results

| Metric                 | Result                                     |
| ---------------------- | ------------------------------------------ |
| Driving success rate   | Approximately 70% (test track environment) |
| Average speed          | Approximately 0.4 m/s                      |
| Training dataset size  | Approximately 3,000 images                 |
| Model parameters       | 1,371,113                                  |

---

## Notes

* Dataset was collected directly using the Raspberry Pi during manual driving.
* Model training was performed on Google Colab using PyTorch.
* The trained `.pth` model file was transferred to the Raspberry Pi for real-time driving tests.
* Regression model provides smoother and more natural steering control compared to classification models.

---

## Future Improvements

Planned enhancements include:

* Obstacle detection and avoidance functionality
* Speed control optimization using reinforcement learning
* Real-time monitoring system
* Real-time camera direction control

---

## References

Raspberry Pi GPIO Programming Guide:
[http://lhdangerous.godohosting.com/wiki/index.php/Raspberry_pi_%ec%97%90%ec%84%9c_python%ec%9c%bc%eb%a1%9c_GPIO_%ec%82%ac%ec%9a%a9%ed%95%98%ea%b8%b0](http://lhdangerous.godohosting.com/wiki/index.php/Raspberry_pi_%ec%97%90%ec%84%9c_python%ec%9c%bc%eb%a1%9c_GPIO_%ec%82%ac%ec%9a%a9%ed%95%98%ea%b8%b0)

DeepThinkCar Mini Project:
[https://github.com/JD-edu/deepThinkCar_mini](https://github.com/JD-edu/deepThinkCar_mini)

---

---

## Measuring the steering error

`evaluate_val.py` loads the trained weights and reports the mean absolute steering error in degrees on a held-out 20% of the frames:

```bash
python evaluate_val.py --data data --weights model_regression.pth
```

Trained weights: TODO (upload the real `model_regression.pth`, or link where it is hosted).
