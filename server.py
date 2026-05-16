import os, time, threading, io
import numpy as np
from flask import Flask, request, jsonify, Response, render_template_string
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

app          = Flask(__name__)
model        = YOLO("yolov8n.pt")
latest_frame = None
lock         = threading.Lock()
stats        = {"total_frames": 0, "total_detections": 0, "fps": 0.0, "avg_inference_ms": 0.0}
fps_history  = []
last_time    = time.time()

COLORS = ["#56B4E9","#E69F00","#009E73","#D55E00","#0072B2","#F0E442","#CC79A7"]

def draw_detections(img_pil, results):
    draw = ImageDraw.Draw(img_pil)
    names = model.names
    detections = []
    for r in results:
        for box in (r.boxes or []):
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = names[cls]
            color = COLORS[cls % len(COLORS)]
            draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
            draw.rectangle([x1, y1-18, x1+len(label)*8+50, y1], fill=color)
            draw.text((x1+2, y1-16), f"{label} {conf:.2f}", fill="white")
            detections.append({"label":label,"confidence":round(conf,4),
                                "bbox":{"x1":x1,"y1":y1,"x2":x2,"y2":y2}})
    return img_pil, detections

@app.route("/detect", methods=["POST"])
def detect():
    global latest_frame, last_time
    data = request.files["frame"].read() if "frame" in request.files else request.data
    if not data:
        return jsonify({"error": "Tidak ada frame"}), 400

    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(img)

    t0      = time.perf_counter()
    results = model(arr, imgsz=320, conf=0.45, iou=0.45, verbose=False)
    inf_ms  = (time.perf_counter() - t0) * 1000

    annotated, detections = draw_detections(img.copy(), results)

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=80)
    with lock:
        latest_frame = buf.getvalue()

    stats["total_frames"]     += 1
    stats["total_detections"] += len(detections)
    now = time.time()
    fps_history.append(now - last_time)
    if len(fps_history) > 30: fps_history.pop(0)
    last_time = now
    avg = sum(fps_history) / len(fps_history)
    stats["fps"] = round(1.0/avg if avg > 0 else 0, 1)
    stats["avg_inference_ms"] = round(inf_ms, 1)

    return jsonify({"status":"ok","count":len(detections),
                    "detections":detections,"inference_ms":round(inf_ms,2),"fps":stats["fps"]})

@app.route("/stream")
def stream():
    def gen():
        while True:
            with lock:
                frame = latest_frame
            if frame is None:
                img = Image.new("RGB",(320,240),(30,30,30))
                ImageDraw.Draw(img).text((60,110),"Menunggu ESP32-CAM...",fill="gray")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                frame = buf.getvalue()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({"status":"running","model":"yolov8n.pt", **stats})

@app.route("/")
def index():
    return render_template_string("""
    <!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">
    <title>ESP32-CAM YOLOv8n</title>
    <style>body{font-family:monospace;background:#111;color:#eee;margin:20px}
    h1{color:#4fc3f7}img{border:2px solid #4fc3f7;border-radius:4px;max-width:100%}
    .s{background:#1e1e1e;padding:12px;border-radius:6px;margin-top:16px}
    .r{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #333}</style>
    <script>async function r(){const d=await(await fetch('/status')).json();
    document.getElementById('fps').textContent=d.fps+' FPS';
    document.getElementById('inf').textContent=d.avg_inference_ms+' ms';
    document.getElementById('fr').textContent=d.total_frames;
    document.getElementById('det').textContent=d.total_detections;}
    setInterval(r,1000);r();</script></head>
    <body><h1>ESP32-CAM + YOLOv8n</h1>
    <img src="/stream"><div class="s">
    <div class="r"><span>FPS</span><span id="fps">-</span></div>
    <div class="r"><span>Inferensi</span><span id="inf">-</span></div>
    <div class="r"><span>Total Frame</span><span id="fr">-</span></div>
    <div class="r"><span>Total Objek</span><span id="det">-</span></div>
    </div></body></html>""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
