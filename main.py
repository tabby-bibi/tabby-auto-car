import cv2
import numpy as np
import time
from picamera2 import Picamera2
from car_controller import CarController

# 관심 영역 ROI 비율 설정
POLY_BOTTOM_WIDTH = 1.5
POLY_TOP_WIDTH = 0.3
POLY_HEIGHT = 0.6

# ROI 마스킹 함수
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[  
        (int(width * (1 - POLY_BOTTOM_WIDTH) / 2), height),
        (int(width * (1 - POLY_TOP_WIDTH) / 2), int(height * (1 - POLY_HEIGHT))),
        (int(width * (1 + POLY_TOP_WIDTH) / 2), int(height * (1 - POLY_HEIGHT))),
        (int(width * (1 + POLY_BOTTOM_WIDTH) / 2), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# 선 분리 함수
def separate_lines(lines, img_center):
    left_lines, right_lines = [], []
    if lines is None:
        return left_lines, right_lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue
        if slope < 0 and x1 < img_center and x2 < img_center:
            left_lines += [[x1, y1], [x2, y2]]
        elif slope > 0 and x1 > img_center and x2 > img_center:
            right_lines += [[x1, y1], [x2, y2]]
    return left_lines, right_lines

def fit_line(points):
    if len(points) == 0:
        return None
    points = np.array(points)
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    slope = vy / vx
    intercept = y - slope * x
    return slope[0], intercept[0]

def make_line_points(y1, y2, slope, intercept):
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def decide_direction(center_lane_x, img_center, left_slope, right_slope,
                     position_thresh=30, slope_thresh=0.6, slope_diff_thresh=0.5):
    diff = center_lane_x - img_center
    if abs(diff) > position_thresh and abs(left_slope) > slope_thresh and abs(right_slope) > slope_thresh and abs(left_slope - right_slope) > slope_diff_thresh:
        return "right" if diff > 0 else "left"
    return "straight"

# 메인 함수
def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    car = CarController()

    try:
        while True:
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            # HSV 흰색 필터링
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 50, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            filtered = cv2.bitwise_and(frame, frame, mask=mask)

            # 엣지 검출
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)

            # ROI 마스킹 (edge만)
            masked = region_of_interest(edges)

            # 직선 검출
            lines = cv2.HoughLinesP(masked, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=30)
            left_points, right_points = separate_lines(lines, img_center)
            left_fit = fit_line(left_points)
            right_fit = fit_line(right_points)

            y1 = height
            y2 = int(height * 0.6)

            if left_fit is not None and right_fit is not None:
                left_slope, left_intercept = left_fit
                right_slope, right_intercept = right_fit
                left_line = make_line_points(y1, y2, left_slope, left_intercept)
                right_line = make_line_points(y1, y2, right_slope, right_intercept)

                if left_line and right_line:
                    center_lane_x = (left_line[0][0] + right_line[0][0]) // 2

                    # 차선 영역 시각화
                    pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]])
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                    cv2.line(frame, left_line[0], left_line[1], (0, 255, 255), 5)
                    cv2.line(frame, right_line[0], right_line[1], (0, 255, 255), 5)

                    direction = decide_direction(center_lane_x, img_center, left_slope, right_slope)
                    if direction == "straight":
                        car.update(direction=None)
                        car.set_motor_forward()
                    elif direction == "right":
                        car.update(direction="right")
                    elif direction == "left":
                        car.update(direction="left")
                    else:
                        car.update(direction=None)

                    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    car.stop_drive()
                    cv2.putText(frame, "No lane detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                car.stop_drive()
                cv2.putText(frame, "No lane detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # 결과 화면 출력
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Edges (masked)", masked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()

# 시작점
if __name__ == "__main__":
    main()
