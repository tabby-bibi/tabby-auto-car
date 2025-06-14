from picamera2 import Picamera2
import cv2
import numpy as np

'''
Edge-based grid lane detection using PiCamera2.
'''

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Determine driving direction based on lane center
def get_direction_text(lane_center, frame_width, threshold=40):
    center_x = frame_width // 2
    diff = lane_center - center_x

    if abs(diff) < threshold:
        return "Straight"
    elif diff < 0:
        return "Left"
    else:
        return "Right"

# Frame processing function
def process_frame(frame):
    height, width = frame.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

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

            roi = edges[y1:y2, x1:x2]
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

    # Determine direction from lane centers
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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        result = process_frame(frame)
        cv2.imshow("Lane Detection", result)
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    picam2.close()
    cv2.destroyAllWindows()
