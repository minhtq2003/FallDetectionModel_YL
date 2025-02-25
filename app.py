from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import cvzone
import math
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import time
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.fx.all import audio_loop

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov88.pt')

# Load class names from classes.txt
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(request.url)

    video = request.files['video']
    if video.filename == '':
        return redirect(request.url)

    # Đảm bảo tên file hợp lệ bằng secure_filename
    filename = secure_filename(video.filename)

    # Lưu video vào thư mục static
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Xử lý video để phát hiện ngã
    output_path, fall_detected = process_video(video_path)

    # Nếu phát hiện ngã, thêm âm thanh vào video
    if fall_detected:
        output_path_with_sound = add_alarm_sound(output_path)
        video_url = url_for('static', filename=os.path.basename(output_path_with_sound))
    else:
        video_url = url_for('static', filename=os.path.basename(output_path))

    # Truyền đường dẫn video đến template result.html
    return render_template('result.html', video_path=video_url)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_video_path = f'static/output_{int(time.time())}.mp4'
    
    # Get video information
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    fall_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = classnames[class_id]
                conf = math.ceil(confidence * 100)

                # Nếu phát hiện người
                if class_name == 'person' and conf > 80:
                    height_box = y2 - y1
                    width_box = x2 - x1
                    ratio = height_box / width_box

                    cvzone.cornerRect(frame, [x1, y1, width_box, height_box], l=30, rt=6)
            
                    # Phát hiện ngã dựa trên tỷ lệ chiều cao/chiều rộng
                    if ratio <0.6:
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                        fall_detected = True
                    else:
                        cvzone.putTextRect(frame, 'Not Fall', [x1, y1 - 20], thickness=2, scale=2)

        # Ghi lại frame vào video kết quả
        out.write(frame)

    cap.release()
    out.release()
    return output_video_path, fall_detected

def add_alarm_sound(video_path):
    """
    Thêm âm thanh báo động vào video nếu phát hiện ngã.
    """
    # Load video và âm thanh báo động
    video = VideoFileClip(video_path)
    alarm_sound = AudioFileClip("static/alarm.mp3")

    # Lặp lại âm thanh để phù hợp với độ dài video
    if alarm_sound.duration < video.duration:
        alarm_sound = audio_loop(alarm_sound, duration=video.duration)

    # Kết hợp video và âm thanh
    final_video = video.set_audio(alarm_sound)

    # Tạo đường dẫn video mới
    final_output_path = video_path.replace('.mp4', '_with_sound.mp4')
    final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

    return final_output_path

def gen_frames():
    cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định

    while True:
        ret, frame = cap.read()  # Đọc frame từ webcam
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = classnames[class_id]
                conf = math.ceil(confidence * 100)

                # Nếu phát hiện người
                if  class_name == 'person' and conf > 60:
                    height_box = y2 - y1
                    width_box = x2 - x1
                    ratio = height_box / width_box

                    cvzone.cornerRect(frame, [x1, y1, width_box, height_box], l=30, rt=6)
            
                    # Phát hiện ngã dựa trên tỷ lệ chiều cao/chiều rộng
                    if 0.5 < ratio <= 0.6:
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    else:
                        cvzone.putTextRect(frame, 'Not Fall', [x1, y1 - 20], thickness=2, scale=2)

        # Mã hóa frame cho luồng MJPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Trả về frame cho luồng video
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    # Trả về frame từ webcam
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
