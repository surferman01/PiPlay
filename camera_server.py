#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, url_for
from picamera2 import Picamera2
from PIL import Image
import io
import time

app = Flask(__name__)

# Initialize camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
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
            max-width: 100%;
            height: auto;
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
    return render_template_string(HTML_PAGE)

def generate_frames():
    """Generator that yields MJPEG frames from the camera."""
    while True:
        # Capture frame as RGB numpy array
        frame = picam2.capture_array()

        # Convert to JPEG using Pillow
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        jpg_bytes = buf.getvalue()

        # Yield frame in multipart/x-mixed-replace format
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n'
        )

        # Limit FPS a bit
        time.sleep(0.05)  # ~20 fps

@app.route("/video_feed")
def video_feed():
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
