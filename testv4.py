import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque


class FallDetectionModel:
    def __init__(self, angle_threshold=70, movement_history=5, confidence_threshold=0.6):
        self.model = YOLO("yolov8n-pose.pt")  # Model YOLO
        self.angle_threshold = angle_threshold
        self.confidence_threshold = confidence_threshold
        self.movement_history = movement_history
        self.keypoints_history = deque(maxlen=movement_history)
        self.timestamps = deque(maxlen=movement_history)
        self.critical_points = [5, 6, 11, 12, 13, 14]  # Vai, hông, đầu gối

    def is_fall_confirmed(self, avg_height, avg_width, delay=2):
        """
        Xác nhận ngã dựa trên tỷ lệ height/width sau khi delay 4 giây (dùng threading).
        """
        def check_fall():
            if avg_width == 0:  # Tránh chia cho 0
                print("Width is zero, skipping fall confirmation.")
                return False
            ratio = avg_height / avg_width
            print(f"Height/Width Ratio: {ratio:.2f}")
            if ratio < 1:
                print("Fall confirmed!")
                return True
            else:
                print("Fall not confirmed.")
                return False

        # Sử dụng threading để trì hoãn kiểm tra
        thread = threading.Thread(target=lambda: (time.sleep(delay), check_fall()))
        thread.start()

    def detect_fall(self, frame, velocity_threshold):
        """
        Phát hiện té ngã trong khung hình.
        """
        results = self.model(frame)
        debug_info = {}

        if len(results) == 0 or results[0].keypoints is None:
            return False, debug_info, None

        keypoints = results[0].keypoints.data[0].cpu().numpy()
        self.keypoints_history.append(keypoints)
        self.timestamps.append(time.time())
        
        vertical_angles = self.calculate_vertical_angles(keypoints)
        mean_angle = np.mean(vertical_angles) if vertical_angles else 90
        velocity = self.calculate_velocity(keypoints)
        
        angle_condition = mean_angle < self.angle_threshold
        velocity_condition = velocity > velocity_threshold
        is_falling = angle_condition and velocity_condition
        
        debug_info = {
            'mean_angle': mean_angle,
            'velocity': velocity,
            'velocity_threshold': velocity_threshold,
            'angle_condition': angle_condition,
            'velocity_condition': velocity_condition
        }
        
        return is_falling, debug_info, keypoints

    # Giữ nguyên các hàm tính toán khác
    def calculate_velocity(self, keypoints):
        if len(self.keypoints_history) < 2:
            return 0

        current_points = np.array(keypoints)
        previous_points = np.array(self.keypoints_history[-2])
        dt = self.timestamps[-1] - self.timestamps[-2]

        if dt == 0:
            return 0

        valid_indices = [
        idx for idx in self.critical_points
        if idx < len(current_points) and idx < len(previous_points) and
          current_points[idx][2] > self.confidence_threshold and
          previous_points[idx][2] > self.confidence_threshold
]


        if not valid_indices:
            return 0

        current_coords = current_points[valid_indices, :2]
        previous_coords = previous_points[valid_indices, :2]
        distances = np.linalg.norm(current_coords - previous_coords, axis=1)
        mean_velocity = np.mean(distances) / dt
        return mean_velocity

    def calculate_vertical_angles(self, keypoints):
        def calculate_angle(p1, p2, p3):
            try:
                if (p1[2] < self.confidence_threshold or
                    p2[2] < self.confidence_threshold or
                    p3[2] < self.confidence_threshold):
                    return None
                ba = np.array(p1[:2]) - np.array(p2[:2])
                bc = np.array(p3[:2]) - np.array(p2[:2])
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cosine_angle))
            except:
                return None

        angles = []
        if len(keypoints) > 14:
            if (keypoints[5][2] > self.confidence_threshold and 
                keypoints[11][2] > self.confidence_threshold and 
                keypoints[13][2] > self.confidence_threshold):
                left_angle = calculate_angle(keypoints[5], keypoints[11], keypoints[13])
                if left_angle is not None:
                    angles.append(left_angle)

            if (keypoints[6][2] > self.confidence_threshold and 
                keypoints[12][2] > self.confidence_threshold and 
                keypoints[14][2] > self.confidence_threshold):
                right_angle = calculate_angle(keypoints[6], keypoints[12], keypoints[14])
                if right_angle is not None:
                    angles.append(right_angle)

        return angles

    def calculate_average_bbox_size(self, results):
        total_width = 0
        total_height = 0
        count = 0

        for result in results:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x_min, y_min, x_max, y_max = box[:4]
                    width = x_max - x_min
                    height = y_max - y_min
                    total_width += width
                    total_height += height
                    count += 1

        if count == 0:
            return 0, 0

        avg_width = total_width / count
        avg_height = total_height / count
        return avg_width, avg_height

    def calculate_velocity_threshold(self, avg_height, scale_factor=0.1):
        return scale_factor * avg_height
    def draw_bounding_box(self, frame, results, color=(0, 255, 0)):
        """
        Vẽ bounding box lên khung hình.
        """
        for result in results:
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    def draw_keypoints(self, frame, keypoints, is_falling,color):
        """
        Vẽ keypoints lên khung hình.
        """
        keypoint_color = (color) 
        for i, keypoint in enumerate(keypoints):
            x, y, confidence = keypoint
            if confidence > self.confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, keypoint_color, -1)
                cv2.putText(frame, str(i), (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, keypoint_color, 1)

def main():
    fall_detector = FallDetectionModel(angle_threshold=70, movement_history=5, confidence_threshold=0.6)
    red = 0,0,255
    blue = 255,0,0
    Color = 0,0,0
    Fall = False

    video_path = "vd5.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        results = fall_detector.model(frame)
        avg_width, avg_height = fall_detector.calculate_average_bbox_size(results)
        

        velocity_threshold = fall_detector.calculate_velocity_threshold(avg_height) if avg_height > 0 else 100
        if Fall == False:
           is_falling, debug_info, keypoints = fall_detector.detect_fall(frame, velocity_threshold)
           fall_detector.draw_bounding_box(frame, results, color=(Color))
        if Fall == True:
         is_falling, debug_info, keypoints = fall_detector.detect_fall(frame, velocity_threshold)
         fall_detector.draw_bounding_box(frame, results, color=(Color))
         is_falling = True
         print(str(is_falling))
        if keypoints is not None:
                  fall_detector.draw_keypoints(frame, keypoints, is_falling,Color)

        if is_falling == True:
            print("Fall detected, checking confirmation...")
            fall_detector.is_fall_confirmed(avg_height, avg_width)
            fall_detector.draw_bounding_box(frame, results, color=(Color))
            Fall = True
        if avg_width != 0:
         if avg_height/avg_width < 1.5 and Fall == True :
            Color = red
         else:
            Color = blue 
            Fall = False
            

        mean_angle = debug_info.get('mean_angle', 90)
        velocity = debug_info.get('velocity', 0)

        cv2.putText(frame, f"Mean Angle: {mean_angle:.2f} degrees", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Mean Velocity: {velocity:.2f} px/s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
