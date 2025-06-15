from picamera2 import Picamera2
import cv2
import numpy as np

'''
그리드 기반 차선 인식 - hsv색공간 기반 흰색/검은색 차선 감지 코드
'''

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Determine driving direction
def get_direction_text(lane_center, frame_width, threshold=40):
    center_x = frame_width // 2
    diff = lane_center - center_x

    if abs(diff) < threshold:
        return "Straight"
    elif diff < 0:
        return "Left"
    else:
        return "Right"

# Process each frame
def process_frame(frame):
    height, width = frame.shape[:2]

    # Preprocessing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White mask
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # Black mask
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # Combine masks
    mask = cv2.bitwise_or(white_mask, black_mask)

    # 명암 보정하는 부분
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # Gaussian blur → edge detection
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Apply mask to edges to keep only white/black lines
    masked_edges = cv2.bitwise_and(edges, mask)

    # Grid parameters
    n_rows = 10
    n_cols = 2
    grid_height = height // n_rows
    grid_width = width // n_cols

    lane_centers = []

    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * grid_width
            y1 = height - (row + 1) * grid_height
            x2 = x1 + grid_width
            y2 = y1 + grid_height

            roi = masked_edges[y1:y2, x1:x2]
            ys, xs = np.where(roi > 0)
            if len(xs) > 0:
                mean_x = int(np.mean(xs))
                color = (0, 0, 255) if col == 0 else (255, 0, 0)
                pt1 = (x1 + mean_x, y2)
                pt2 = (x1 + mean_x, y1)
                cv2.line(frame, pt1, pt2, color, 2)
                lane_centers.append(x1 + mean_x)

            # Draw grid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Show direction
    if lane_centers:
        lane_center = int(np.mean(lane_centers))
        cv2.line(frame, (lane_center, height), (lane_center, height - 50), (0, 255, 255), 3)
        direction = get_direction_text(lane_center, width)
        cv2.putText(frame, f"Direction: {direction}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        cv2.putText(frame, "No lane detected", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return frame

# Real-time loop
try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = process_frame(frame)
        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    picam2.close()
    cv2.destroyAllWindows()
