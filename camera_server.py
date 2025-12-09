#!/usr/bin/env python3
from flask import Flask, Response, render_template_string, url_for, jsonify
import threading
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
    main={"size": (1280, 720), "format": "RGB888"}  # same as your config
)
picam2.configure(camera_config)
picam2.start()

# -----------------------------
# YOLO model setup
# -----------------------------
model = YOLO("yolov8n.pt")  # same model as before

# Find the class ID for "bottle" in this model
bottle_class_id = None
for k, v in model.names.items():
    if v == "bottle":
        bottle_class_id = int(k)
        break

if bottle_class_id is None:
    raise RuntimeError("This YOLO model does not have a 'bottle' class.")

print(f"[INIT] Bottle class id: {bottle_class_id}")

# Shared state for bottle detection
bottle_last_seen = 0.0
bottle_lock = threading.Lock()

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
            margin: 0;
            padding: 0;
        }
        .wrapper {
            padding: 20px;
        }
        h1 {
            margin-bottom: 10px;
        }
        p {
            margin-top: 0;
            margin-bottom: 20px;
        }
        img {
            border: 4px solid #444;
            display: block;
            margin: 0 auto;
            width: 90vw;       /* big video */
            max-width: 1280px;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <h1>Raspberry Pi YOLO Object Detection</h1>
        <p>
            If you don't see video, give it a few seconds or refresh.
            For sound: most browsers need you to click anywhere on the page once
            before audio is allowed to play.
        </p>

        <img src="{{ url_for('video_feed') }}" />

        <!-- Audio element for bottle sound -->
        <audio id="bottleSound" src="{{ url_for('static', filename='fart-03.mp3') }}"></audio>
    </div>

    <script>
    let lastBottleTrigger = 0;
    const audioEl = document.getElementById('bottleSound');

    // Basic logging for audio load
    audioEl.addEventListener('canplaythrough', () => {
        console.log('[CLIENT] Audio loaded and ready.');
    });
    audioEl.addEventListener('error', (e) => {
        console.error('[CLIENT] Error loading fart-03.mp3:', e);
    });

    async function checkBottle() {
        try {
            const res = await fetch("{{ url_for('detection_status') }}");
            const data = await res.json();
            const now = Date.now();

            if (data.bottle) {
                // Throttle sound: max once every 3 seconds
                if (now - lastBottleTrigger > 3000) {
                    console.log('[CLIENT] Bottle active; trying to play fart sound.');
                    audioEl.currentTime = 0;
                    audioEl.play().then(() => {
                        console.log('[CLIENT] Fart sound PLAYED');
                    }).catch(err => {
                        console.log('[CLIENT] Audio play blocked or failed:', err);
                    });
                    lastBottleTrigger = now;
                }
            }
        } catch (e) {
            console.error('[CLIENT] Error checking bottle status:', e);
        }
    }

    // Poll the server every 500ms
    setInterval(checkBottle, 500);
    </script>
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
    global bottle_last_seen  # ensure we update the shared variable

    frame_count = 0
    last_results = None

    # Optional: try to load a font, but fall back gracefully
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    while True:
        try:
            # Capture frame as RGB numpy array (H, W, 3)
            frame = picam2.capture_array()

            # Run YOLO every N frames to save CPU
            N = 8
            if frame_count % N == 0:
                print(f"[YOLO] Running YOLO on frame {frame_count}")
                results = model(frame, imgsz=320, conf=0.5, verbose=False)
                last_results = results[0]  # keep latest result
            frame_count += 1

            # Convert to PIL image for drawing
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # Draw detections if we have results
            if last_results is not None and last_results.boxes is not None:
                boxes = last_results.boxes

                any_bottle = False

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    label_name = model.names.get(cls_id, str(cls_id))
                    label = f"{label_name} {conf:.2f}"

                    # Check for bottle with >= 0.5 confidence
                    if cls_id == bottle_class_id and conf >= 0.5:
                        any_bottle = True
                        print(f"[YOLO] BOTTLE detected! conf={conf:.2f} at box=({x1},{y1},{x2},{y2})")

                    # Draw rectangle and simple label (no fancy bg to avoid errors)
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                    draw.text((x1 + 2, y1 + 2), label, font=font, fill=(0, 255, 0))

                # If any bottle detected in this frame, update last_seen time
                if any_bottle:
                    with bottle_lock:
                        bottle_last_seen = time.time()
                    print(f"[YOLO] bottle_last_seen updated to {bottle_last_seen}")

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

        except Exception as e:
            # Log any error inside the generator so it doesn't fail silently
            print("[ERROR] Exception in generate_frames loop:", repr(e))
            time.sleep(0.5)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/detection_status")
def detection_status():
    """Return JSON telling if a bottle was seen very recently."""
    with bottle_lock:
        last = bottle_last_seen
    age = time.time() - last
    active = age < 1.5
    print(f"[STATUS] detection_status: active={active}, last_seen={last}, age={age:.3f}s")
    return jsonify({"bottle": active})

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("[INIT] Server starting on 0.0.0.0:5000")
    print("[INIT] Ensure static/fart-03.mp3 exists.")
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
    finally:
        picam2.stop()
        print("[SHUTDOWN] Camera stopped.")
