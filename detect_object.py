import time
import board
import busio
import adafruit_vl53l0x

SAFE_DISTANCE = 200 # 전진 기준 거리 : 20센티
STOP_DISTANCE = 100 # 중지 기준 거리 : 10센티

class ObjectDetector:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c)
       # self.sensor.measurement_timing_budget = 200000
    def get_distance(self):
        ''' 현재 장애물과의 거리를 반환하는 함수입니다.'''
        return self.sensor.range

    def check_if_object_detected(self):
        distance = self.get_distance()
        if distance > SAFE_DISTANCE:
            return "clear"
        elif distance < STOP_DISTANCE:
            return "stop"
        else:
            return "warning"

