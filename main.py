import cv2
import numpy as np
import time
import math
from picamera2 import Picamera2
from car_controller import CarController

POLY_BOTTOM_WIDTH = 2.0
POLY_TOP_WIDTH = 0.4
POLY_HEIGHT = 0.3

def filter_white(img):
    lower = np.array([150, 150, 150])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)
    return cv2.bitwise_and(img, img, mask=mask)

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

def separate_lines(lines, img_center):
    left_lines = []
    right_lines = []
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
            left_lines.extend([[x1, y1], [x2, y2]])
        elif slope > 0 and x1 > img_center and x2 > img_center:
            right_lines.extend([[x1, y1], [x2, y2]])
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

def decide_direction_by_slope(left_slope, right_slope,
                               angle_thresh=30, slope_diff_thresh=0.5):
    left_angle = math.degrees(math.atan(left_slope))
    right_angle = math.degrees(math.atan(right_slope))
    avg_angle = (left_angle + right_angle) / 2
    slope_diff = abs(left_slope - right_slope)

    if abs(avg_angle) < angle_thresh and slope_diff < slope_diff_thresh:
        return "straight"
    elif avg_angle < -angle_thresh:
        return "left"
    elif avg_angle > angle_thresh:
        return "right"
    else:
        return "straight"

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

            filtered = filter_white(frame)
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 100)
            masked = region_of_interest(edges)

            lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=50)
            left_points, right_points = separate_lines(lines, img_center)
            left_fit = fit_line(left_points)
            right_fit = fit_line(right_points)

            y1 = height
            y2 = int(height * 0.6)

            if left_fit and right_fit:
                left_slope, left_intercept = left_fit
                right_slope, right_intercept = right_fit

                left_line = make_line_points(y1, y2, left_slope, left_intercept)
                right_line = make_line_points(y1, y2, right_slope, right_intercept)

                if left_line and right_line:
                    center_lane_x = (left_line[0][0] + right_line[0][0]) // 2

                    pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]])
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                    cv2.line(frame, left_line[0], left_line[1], (0, 255, 255), 5)
                    cv2.line(frame, right_line[0], right_line[1], (0, 255, 255), 5)

                    direction = decide_direction_by_slope(left_slope, right_slope)

                    if direction == "straight":
                        car.update(direction=None)
                        car.set_motor_forward()
                        
                    elif direction == "left":
                        car.update(direction="left")
                        
                    elif direction == "right":
                        car.update(direction="right")
                        
                    else:
                        car.update(direction=None)
                        car.stop_drive()

                    cv2.putText(frame, f"Direction: {direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    car.stop_drive()
                    cv2.putText(frame, "No lane detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                car.stop_drive()
                cv2.putText(frame, "No lane detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Lane Detection", frame)
            cv2.imshow("Edges", edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
