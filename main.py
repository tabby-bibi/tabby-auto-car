from lane_follower import LaneFollower
from car_controller import CarController
import time
import cv2

def main():
    lane_detector = LaneFollower()
    car = CarController(frame_width=320)  # 프레임 크기는 카메라에 맞게 조정

    try:
        while True:
            ret, frame = lane_detector.cap.read()
            if not ret:
                print("카메라에서 프레임을 가져올 수 없습니다.")
                break

            processed = lane_detector.process_frame(frame)
            lane_center, frame_center = lane_detector.find_lane_center(processed)

            # 차량 제어 업데이트
            car.update(lane_center)

            # 디버깅용 영상 출력
            cv2.line(frame, (lane_center, 0), (lane_center, frame.shape[0]), (0, 255, 0), 2)
            cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (255, 0, 0), 1)
            cv2.imshow("Camera", frame)
            cv2.imshow("Processed", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("사용자 중지 감지. 종료합니다.")

    finally:
        lane_detector.cap.release()
        cv2.destroyAllWindows()
        car.cleanup()

if __name__ == "__main__":
    main()
