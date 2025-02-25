from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import cvzone
import math
import os
import tensorflow as tf
import torch as torch
import time as timer
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
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            # If the file doesn't exist, create a new one and add column headers
            if not os.path.exists(excel_file_path):
                workbook = openpyxl.Workbook()
                sheet = workbook.active
                sheet.title = "Fall Events"
                sheet.append(["Video Path", "Fall Time"])
            else:
                try:
                    workbook = openpyxl.load_workbook(excel_file_path)
                except ReadOnlyWorkbookException:
                    print(f"Warning: Excel file is read-only. Skipping log entry for {video_path}")
                    return

            sheet = workbook.active

            # Add data to a new row
            sheet.append([video_path, fall_time])

            # Save the file
            workbook.save(excel_file_path)
            workbook.close()
            print(f"Fall event saved: {video_path} at {fall_time}")
            return  # Successfully saved, exit the function

        except PermissionError:
            print(f"Permission error when trying to save to Excel. Attempt {attempt + 1} of {max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"Failed to save fall event for {video_path} after {max_retries} attempts.")
        except Exception as e:
            print(f"An error occurred while saving to Excel: {str(e)}")
            break  # For other exceptions, we break immediately

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_video_path = f'static/output_{int(time.time())}.mp4'

    # Get video information
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    fall_detected = False
    last_fall_time = 0  # Time of the last fall detection
    cooldown_period = 10  # Cooldown period of 10 seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

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

                # If a person is detected
                if class_name == 'person' and conf > 80:
                    height_box = y2 - y1
                    width_box = x2 - x1
                    ratio = height_box / width_box

                    cvzone.cornerRect(frame, [x1, y1, width_box, height_box], l=30, rt=6)

                    # Detect fall based on height/width ratio
                    if ratio < 0.6:
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                        
                        # Check if enough cooldown time has passed since the last fall detection
                        if current_time - last_fall_time > cooldown_period:
                            fall_detected = True
                            fall_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                            save_fall_event_to_excel(fall_time, video_path)
                            last_fall_time = current_time
                    else:
                        cvzone.putTextRect(frame, 'Not Fall', [x1, y1 - 20], thickness=2, scale=2)

        # Write the frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    return output_video_path, fall_detected

def add_alarm_sound(video_path):
    """
    Add alarm sound to the video if a fall is detected.
    """
    # Load video and alarm sound
    video = VideoFileClip(video_path)
    alarm_sound = AudioFileClip("static/alarm.mp3")

    # Loop the sound to match the video duration
    if alarm_sound.duration < video.duration:
        alarm_sound = audio_loop(alarm_sound, duration=video.duration)

    # Combine video and sound
    final_video = video.set_audio(alarm_sound)

    # Create new video path
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

    # Ensure the filename is secure
    filename = secure_filename(video.filename)

    # Save the video to the static folder
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Process the video to detect falls
    output_path, fall_detected = process_video(video_path)

    # If a fall is detected, add sound to the video
    if fall_detected:
        output_path_with_sound = add_alarm_sound(output_path)
        video_url = url_for('static', filename=os.path.basename(output_path_with_sound))
    else:
        video_url = url_for('static', filename=os.path.basename(output_path))

    # Pass the video path to the result.html template
    return render_template('result.html', video_path=video_url)

@app.route('/video_feed')
def video_feed():
    # Return frames from the webcam
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use default webcam
    last_fall_time = 0
    cooldown_period = 10  # Cooldown period of 10 seconds

    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        current_time = time.time()

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

                # If a person is detected
                if class_name == 'person' and conf > 60:
                    height_box = y2 - y1
                    width_box = x2 - x1
                    ratio = height_box / width_box

                    cvzone.cornerRect(frame, [x1, y1, width_box, height_box], l=30, rt=6)

                    # Detect fall based on height/width ratio
                    if ratio < 0.6:
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                        # Check if enough cooldown time has passed since the last fall detection
                        if current_time - last_fall_time > cooldown_period:
                            fall_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                            save_fall_event_to_excel(fall_time, "live_stream")
                            last_fall_time = current_time
                    else:
                        cvzone.putTextRect(frame, 'Not Fall', [x1, y1 - 20], thickness=2, scale=2)

        # Encode frame for MJPEG stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return frame for video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == "__main__":
    app.run(debug=True)