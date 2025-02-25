import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time


class FallDetectionModel:
    def __init__(self, angle_threshold=70, movement_history=5, confidence_threshold=0.5):
        self.model = YOLO("yolov8n-pose.pt")  # Model YOLO
        self.angle_threshold = angle_threshold
        self.confidence_threshold = confidence_threshold
        self.movement_history = movement_history
        self.keypoints_history = deque(maxlen=movement_history)
        self.timestamps = deque(maxlen=movement_history)
        self.critical_points = [5, 6, 11, 12, 13, 14]  # Vai, hông, đầu gối

    def calculate_velocity(self, keypoints):
       try:
        # Nếu không đủ lịch sử keypoints, trả về 0
         if len(self.keypoints_history) < 2:
            return 0.0

        # Chuyển đổi keypoints sang numpy array
         current_points = np.array(keypoints)
         previous_points = np.array(self.keypoints_history[-1])

        # Tập trung vào các điểm quan trọng (vai, hông, đầu gối)
         critical_indices = [5, 6, 11, 12, 13, 14]

        # Lọc các điểm có độ tin cậy cao
         valid_points = [
            idx for idx in critical_indices 
            if (current_points[idx][2] > self.confidence_threshold and 
                previous_points[idx][2] > self.confidence_threshold)
        ]

        # Nếu không có điểm nào hợp lệ, trả về 0
         if not valid_points:
            return 0.0

        # Tính khoảng cách di chuyển của các điểm
         distances = []
         for idx in valid_points:
            current_point = current_points[idx][:2]
            previous_point = previous_points[idx][:2]
            distance = np.linalg.norm(current_point - previous_point)
            distances.append(distance)

        # Tính thời gian giữa hai khung hình
         dt = self.timestamps[-1] - self.timestamps[-2] if len(self.timestamps) >= 2 else 1

        # Tính vận tốc trung bình
         mean_velocity = np.mean(distances) / max(dt, 0.001)

         return float(mean_velocity)

       except Exception as e:
         print(f"Lỗi khi tính vận tốc: {e}")
         return 0.0

    def calculate_vertical_angles(self, keypoints):
        """
        Tính các góc nghiêng theo phương thẳng đứng.
        """
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
    def detect_fall(self, frame, velocity_threshold):

     results = self.model(frame)
     debug_info = {}

     if len(results) == 0 or results[0].keypoints is None:
         return False, debug_info, None

     keypoints = results[0].keypoints.data[0].cpu().numpy()
     self.keypoints_history.append(keypoints)
     self.timestamps.append(time.time())
    
     vertical_angles = self.calculate_vertical_angles(keypoints)
     mean_angle = np.mean(vertical_angles) if vertical_angles else 90
     velocity = self.calculate_velocity(keypoints)  # Tính vận tốc trung bình
    
     angle_condition = mean_angle > self.angle_threshold
     velocity_condition = velocity > velocity_threshold
     is_falling = angle_condition and velocity_condition
    
     debug_info = {
        'mean_angle': mean_angle,
        'velocity': velocity,  # Thêm vận tốc vào debug_info
        'velocity_threshold': velocity_threshold,
        'angle_condition': angle_condition,
        'velocity_condition': velocity_condition
     }
    
     return is_falling, debug_info, keypoints

    def calculate_average_bbox_size(self, results):
        """
        Tính trung bình kích thước bounding box.
        """
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
        """
        Tính velocity_threshold dựa trên chiều cao bounding box.
        """
        return scale_factor * avg_height


def main():
    fall_detector = FallDetectionModel(angle_threshold=70, movement_history=5, confidence_threshold=0.5)

    # Đường dẫn tới file video
    video_path = "fall.mp4"
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        # Dự đoán bounding boxes và keypoints
        results = fall_detector.model(frame)
        avg_width, avg_height = fall_detector.calculate_average_bbox_size(results)

        # Tính velocity_threshold dựa trên bounding box
        velocity_threshold = fall_detector.calculate_velocity_threshold(avg_height) if avg_height > 0 else 100

        # ngu thi nga khoc loc cl
        is_falling, debug_info, keypoints = fall_detector.detect_fall(frame, velocity_threshold)

        # Hiển thị thông tin lên frame và in ra terminal
        if keypoints is not None:
            keypoint_color = (0, 0, 255) if is_falling else (0, 255, 0)
            for point in keypoints:
                x, y, conf = point
                if conf > fall_detector.confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 4, keypoint_color, -1)

            if is_falling:
                print("\n=== Fall Detected ===")
                for k, v in debug_info.items():
                    print(f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")
                print("=====================\n")

        # Lấy giá trị mean_angle và velocity để hiển thị lên khung hình
        mean_angle = debug_info.get('mean_angle', 90)
        velocity = debug_info.get('velocity', 0)  # Giá trị vận tốc trung bình (mean velocity)

        # Hiển thị thông tin lên khung hình
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
