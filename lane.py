import cv2
import numpy as np
from picamera2 import Picamera2
import time

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (0, height // 2),
        (width, height // 2),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)
    return roi

def sliding_window(binary):
    hist = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = len(hist) // 2
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    nwindows = 9
    margin = 30  # 해상도 줄이면 마진도 줄이자
    minpix = 20
    window_height = binary.shape[0] // nwindows

    nonzero = binary.nonzero()
    nonzeroy, nonzerox = nonzero[0], nonzero[1]
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0 and len(rightx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        center_fitx = (left_fitx + right_fitx) / 2

        return ploty, left_fitx, right_fitx, center_fitx
    else:
        return None, None, None, None

# --------- PiCamera2 초기화 ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
picam2.configure(config)
picam2.start()
time.sleep(1)

# --------- 메인 루프 ----------
while True:
    frame = picam2.capture_array()
    binary = preprocess(frame)
    ploty, left_fitx, right_fitx, center_fitx = sliding_window(binary)

    result = frame.copy()

    if center_fitx is not None:
        for i in range(len(ploty)):
            cv2.circle(result, (int(left_fitx[i]), int(ploty[i])), 1, (255, 0, 0), -1)
            cv2.circle(result, (int(right_fitx[i]), int(ploty[i])), 1, (0, 0, 255), -1)
            cv2.circle(result, (int(center_fitx[i]), int(ploty[i])), 1, (0, 255, 0), -1)

        lane_center = center_fitx[-1]
        frame_center = frame.shape[1] // 2
        offset = frame_center - lane_center

        cv2.putText(result, f"Offset: {offset:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Lane Detection Realtime", result)
    if cv2.waitKey(10) & 0xFF == 27:  # ESC 키 종료
        break

cv2.destroyAllWindows()
picam2.stop()
