import cv2
import numpy as np
from picamera2 import Picamera2
from car_controller import CarController

'''
Custom camera setup to avoid exposure issues on Raspberry Pi camera.
This script uses manual exposure and white balance settings.
'''

# Set Region of Interest (ROI)
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (0, int(height * 0.5)),
        (width, int(height * 0.5)),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# Compute histogram of pixel intensities along the x-axis
def compute_histogram(img):
    return np.sum(img, axis=0)

# Visualize histogram as an image
def draw_histogram(hist):
    hist_img = np.zeros((200, 640, 3), dtype=np.uint8)
    max_val = np.max(hist)
    if max_val == 0:
        return hist_img

    hist_norm = (hist / max_val) * 200  # Normalize
    for x in range(len(hist)):
        cv2.line(hist_img, (x, 200), (x, 200 - int(hist_norm[x])), (255, 255, 255), 1)
    return hist_img

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)},
        controls={"FrameRate": 15}
    )
    picam2.configure(config)

    # Manual exposure and white balance
    picam2.set_controls({
        "AwbEnable": False,
        "AeEnable": False,
        "ExposureTime": 10000,       # 10ms
        "AnalogueGain": 1.0,
        "ColourGains": (1.5, 1.5)
    })

    picam2.start()
    car = CarController()

    try:
        while True:
            print("\n[Frame captured]")
            frame = picam2.capture_array()
            height, width = frame.shape[:2]
            img_center = width // 2

            # Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Converted to grayscale")
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            print("Applied Gaussian blur")
            edges = cv2.Canny(blur, 30, 120)
            print("Detected Canny edges")

            roi = region_of_interest(edges)
            print("Extracted region of interest")

            # Split into left and right regions
            left_roi = roi[:, :img_center]
            right_roi = roi[:, img_center:]

            left_count = cv2.countNonZero(left_roi)
            right_count = cv2.countNonZero(right_roi)
            print(f"Edge pixels - Left: {left_count}, Right: {right_count}")

            left_hist = compute_histogram(left_roi)
            right_hist = compute_histogram(right_roi)

            if np.max(left_hist) > 0 and np.max(right_hist) > 0:
                left_center = np.argmax(left_hist)
                right_center = np.argmax(right_hist) + img_center
                lane_center = (left_center + right_center) // 2
                print(f"Left peak: {left_center}, Right peak: {right_center}, Lane center: {lane_center}")
            else:
                full_hist = compute_histogram(roi)
                lane_center = np.argmax(full_hist)
                print("Only one side detected. Lane center (full ROI):", lane_center)

            offset = lane_center - img_center
            print(f"Image center: {img_center}, Offset: {offset}")

            threshold = 30
            if left_count == 0 and right_count == 0:
                print("No lane detected. Stopping.")
                direction = "stop"
                car.update(direction=None)
                car.stop_drive()

            elif abs(offset) < threshold:
                if abs(left_count - right_count) > 4000:
                    direction = "right" if left_count > right_count else "left"
                    print(f"Balanced offset, but edge imbalance detected. Adjusting to the {direction}.")
                else:
                    direction = "straight"
                    print("Going straight.")

                if direction == "straight":
                    car.update(direction=None)
                    car.set_motor_forward()
                else:
                    car.update(direction)

            else:
                direction = "right" if offset < -threshold else "left"
                print(f"Large offset detected. Turning {direction}.")
                car.update(direction)

            # Visualization
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.line(frame, (lane_center, height), (lane_center, int(height * 0.6)), (255, 0, 0), 2)
            cv2.line(frame, (img_center, height), (img_center, int(height * 0.6)), (0, 255, 0), 2)

            cv2.imshow("Lane Tracking", frame)
            cv2.imshow("ROI Edge", roi)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")

    finally:
        car.stop_drive()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
