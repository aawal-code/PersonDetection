from flask import Flask, render_template, request, redirect, url_for
import threading
import cv2
import os
import datetime
import subprocess
import webbrowser
import time

app = Flask(__name__)

detection_running = False

# LOGGING
def log_event(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {message}\n"
    with open("logs/event_log.txt", "a") as log_file:
        log_file.write(log_message)
    print(log_message)  # Optional: print log message to console for real-time feedback

# CAPTURES
def capture_image(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"captures/person_detected_{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    log_event(f"Image captured: {image_path}")

# OPEN WEBPAGE
def open_webpage(url):
    try:
        webbrowser.open(url)
        log_event(f"Opened webpage: {url}")
    except Exception as e:
        log_event(f"Failed to open webpage: {str(e)}")

# DETECTION
def detect_person(flag):
    global detection_running
    video_capture = None
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
        if not video_capture.isOpened():
            raise Exception("Could not open video device")

        while detection_running:
            ret, frame = video_capture.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (320, 240))  # Resize frame for faster processing
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                capture_image(frame)
                log_event("Person detected.")
                if not flag['opened']:
                    open_webpage("https://www.classroom.google.com")  # Replace with your desired URL
                    flag['opened'] = True
            else:
                log_event("No person detected.")

            time.sleep(20)  # Wait for 20 seconds before repeating
    except Exception as e:
        log_event(f"Error: {str(e)}")
    finally:
        if video_capture:
            video_capture.release()

# RUNNING DETECTION
def run_detection():
    flag = {'opened': False}
    while detection_running:
        detect_person(flag)

@app.route('/')
def index():
    global detection_running
    return render_template('index.html', detection_running=detection_running)

@app.route('/start')
def start():
    global detection_running
    if not detection_running:
        detection_running = True
        thread = threading.Thread(target=run_detection)
        thread.start()
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global detection_running
    detection_running = False
    log_event("Detection stopped.")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
