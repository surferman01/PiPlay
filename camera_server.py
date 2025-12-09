#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, url_for
from picamera2 import Picamera2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import time
import numpy as np

app = Flask(__name__)

# -----------------------------
# Camera setup
# -----------------------------
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}  # keep it modest for speed
)
picam2.configure(camera_config)
picam2.start()

# -----------------------------
# YOLO model setup
# -----------------------------
# Use a small model to keep it (semi) real-time on a Pi.
# You can download 'yolov8n.pt' beforehand or let ultralytics fetch it.
model = YOLO("yolov8n.pt")  # change path if using a custom model

# -----------------------------
# HTML
# -----------------------------
HTML_PAGE = """
<!doctype html>
<html>
<head>
    <title>Raspberry Pi YOLO Stream</title>
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
    <h1>Raspberry Pi YOLO Object Detection</h1>
    <p>If you don't see video, give it a few seconds or refresh.</p>
    <img src="{{ url_for('video_feed') }}" />
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# -----------------------------
# Frame generator with YOLO
# -----------------------------
def generate_frames():
    """
    Capture frames from the Pi camera, run YOLO detection,
    draw boxes & labels, and stream as MJPEG.
    """
    frame_count = 0
    last_results = None

    # Optional: try to load a font, but fall back gracefully
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    while True:
        # Capture frame as RGB numpy array (H, W, 3)
        frame = picam2.capture_array()

        # Optionally downscale for speed (uncomment if needed)
        # frame = frame[::2, ::2, :]  # simple 1/2 downscale

        # Run YOLO every N frames to save CPU
        # Set N=1 to run on every frame (slower).
        N = 2
        if frame_count % N == 0:
            # YOLO expects numpy arrays; it can handle RGB just fine.
            # You can tweak imgsz and conf to your needs.
            results = model(frame, imgsz=640, conf=0.5, verbose=False)
            last_results = results[0]  # keep latest result
        frame_count += 1

        # Convert to PIL image for drawing
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Draw detections if we have results
        if last_results is not None and last_results.boxes is not None:
            boxes = last_results.boxes

            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Class & confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names.get(cls_id, str(cls_id))} {conf:.2f}"

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

                # Draw label background
                text_size = draw.textbbox((0, 0), label, font=font)
                text_w = text_size[2] - text_size[0]
                text_h = text_size[3] - text_size[1]
                text_bg = [x1, y1 - text_h - 4, x1 + text_w + 4, y1]
                draw.rectangle(text_bg, fill=(0, 255, 0))

                # Draw label text
                draw.text((x1 + 2, y1 - text_h - 2), label, font=font, fill=(0, 0, 0))

        # Encode to JPEG
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        jpg_bytes = buf.getvalue()

        # Yield as MJPEG frame
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n'
        )

        # Small sleep to avoid pinning CPU at 100%
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        picam2.stop()
