import cv2

class VideoCamera:
    def __init__(self, video_source=0):
        # Nếu video_source là đường dẫn đến video thì mở video đó, nếu không sẽ mở camera
        self.video = cv2.VideoCapture(video_source)

    def __del__(self):
        # Giải phóng tài nguyên camera hoặc video khi không còn sử dụng
        self.video.release()

    def get_frame(self):
        # Đọc frame từ video
        success, image = self.video.read()
        if not success:
            return None

        # Chuyển đổi hình ảnh từ BGR sang định dạng JPEG để trả về cho Flask
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
