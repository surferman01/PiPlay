#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import threading
import time

app = Flask(__name__)

# Initialize camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Simple HTML page
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>Raspberry Pi Camera</title>
    <style>
        body {
            background: #222;
            color: #eee;
            font-family: sans-serif;
            text-align: center;
        }
        img {
            border: 4px solid #444;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Raspberry Pi Camera Stream</h1>
    <p>If you don't see video, give it a few seconds or refresh.</p>
    <img src="{{ url_for('video_feed') }}" />
</body>
</html>
"""

@app.route("/")
def index():
    # Main page
    return render_template_string(HTML_PAGE)

def generate_frames():
    """Generator that yields MJPEG frames from the camera."""
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()

        # Optional: convert color for consistency (OpenCV expects BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        jpg_bytes = buffer.tobytes()

        # Yield frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

        # Sleep a bit to avoid maxing out CPU; adjust for desired FPS
        time.sleep(0.05)  # ~20 fps

@app.route("/video_feed")
def video_feed():
    # Video streaming route
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    try:
        # host="0.0.0.0" allows access from other devices on your network
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        picam2.stop()
