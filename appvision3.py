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
import openpyxl
from openpyxl.utils.exceptions import ReadOnlyWorkbookException

app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load class names from classes.txt
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'static'

# Path to the Excel file for saving fall detection times
excel_file_path = 'fall_detection_log.xlsx'

def save_fall_event_to_excel(fall_time, video_path):
    """
    Save fall detection information to Excel file with error handling.
    """
    try:
        if not os.path.exists(excel_file_path):
            # Create Excel file if it doesn't exist
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Fall Events"
            sheet.append(["Video Path", "Fall Time"])
            workbook.save(excel_file_path)

        # Load workbook and append data
        workbook = openpyxl.load_workbook(excel_file_path)
        sheet = workbook.active
        sheet.append([video_path, fall_time])
        workbook.save(excel_file_path)
        workbook.close()
        print(f"Fall event saved: {video_path} at {fall_time}")
    except PermissionError:
        print(f"Permission error: Unable to save fall event for {video_path}")
    except Exception as e:
        print(f"Error while saving fall event: {str(e)}")


def process_video(video_path):
    """
    Process a video to detect falls and save the results.
    """
    cap = cv2.VideoCapture(video_path)
    output_video_path = f'static/output_{int(time.time())}.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    fall_detected = False
    last_fall_time = 0
    cooldown_period = 10  # Cooldown period of 10 seconds
    prev_centers = {}
    velocity_threshold = 50

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with YOLO
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

                # Detect persons
                if class_name == 'person' and conf > 80:
                    height_box = y2 - y1
                    width_box = x2 - x1
                    ratio = height_box / width_box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    cvzone.cornerRect(frame, [x1, y1, width_box, height_box], l=30, rt=6)

                    # Calculate velocity
                    if class_id in prev_centers:
                        prev_center = prev_centers[class_id]
                        velocity = math.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                        if velocity > velocity_threshold and ratio < 0.6:
                            cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                            if time.time() - last_fall_time > cooldown_period:
                                fall_detected = True
                                fall_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                                save_fall_event_to_excel(fall_time, video_path)
                                last_fall_time = time.time()
                    else:
                        velocity = 0

                    prev_centers[class_id] = center

        out.write(frame)

    cap.release()
    out.release()
    return output_video_path, fall_detected


def add_alarm_sound(video_path):
    """
    Add alarm sound to the video if a fall is detected.
    """
    video = VideoFileClip(video_path)
    alarm_sound = AudioFileClip("static/alarm.mp3")

    if alarm_sound.duration < video.duration:
        alarm_sound = audio_loop(alarm_sound, duration=video.duration)

    final_video = video.set_audio(alarm_sound)
    final_output_path = video_path.replace('.mp4', '_with_sound.mp4')
    final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

    return final_output_path


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

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    output_path, fall_detected = process_video(video_path)

    if fall_detected:
        output_path_with_sound = add_alarm_sound(output_path)
        video_url = url_for('static', filename=os.path.basename(output_path_with_sound))
    else:
        video_url = url_for('static', filename=os.path.basename(output_path))

    return render_template('result.html', video_path=video_url)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    cap = cv2.VideoCapture(0)
    prev_centers = {}
    velocity_threshold = 50
    cooldown_period = 10
    last_fall_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip frame horizontally if needed
        results = model(frame)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = classnames[class_id]

                if class_name == 'person' and confidence > 0.8:
                    cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


if __name__ == "__main__":
    app.run(debug=True)
