import cv2
import os
import datetime
import subprocess
import webbrowser
import time

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
def detect_person():
    video_capture = None
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
        if not video_capture.isOpened():
            raise Exception("Could not open video device")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (320, 240))  # Resize frame for faster processing
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                capture_image(frame)
                open_webpage("https://www.classroom.google.com")  # Replace with your desired URL

            time.sleep(20)  # Wait for 20 seconds before repeating
    except Exception as e:
        log_event(f"Error: {str(e)}")
    finally:
        if video_capture:
            video_capture.release()
            # cv2.destroyAllWindows()  # Commented out for headless operation

# RUNNING DETECTION
if __name__ == "__main__":
    log_event("Person detection started.")
    while True:
        detect_person()