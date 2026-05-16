"""
Python Flask Server — YOLOv8n Object Detection
===============================================
Menerima frame JPEG dari ESP32-CAM,
menjalankan inferensi YOLOv8n, dan mengembalikan
hasil deteksi dalam format JSON.

Instalasi dependensi:
    pip install flask ultralytics opencv-python numpy

Jalankan:
    python server.py

Endpoint:
    POST /detect    → terima frame, kembalikan JSON deteksi
    GET  /stream    → MJPEG stream dengan bounding box
    GET  /status    → status server dan statistik
"""

import cv2
import numpy as np
import base64
import time
import threading
from collections import deque
from flask import Flask, request, jsonify, Response, render_template_string
from ultralytics import YOLO

# ─── Konfigurasi ───────────────────────────────────────────────────────────
MODEL_PATH       = "yolov8n.pt"   # Akan diunduh otomatis jika belum ada
CONFIDENCE_THRES = 0.45           # Ambang batas confidence (0.0–1.0)
IOU_THRES        = 0.45           # IoU untuk NMS
IMG_SIZE         = 320            # Ukuran input model (kelipatan 32)
SERVER_HOST      = "0.0.0.0"     # Dengarkan semua antarmuka jaringan
SERVER_PORT      = 5000

# Warna per kelas (BGR) — gunakan palet berbeda agar mudah dibedakan
COLORS = [
    (86, 180, 233), (230, 159, 0), (0, 158, 115),
    (213, 94, 0),   (0, 114, 178), (240, 228, 66),
    (204, 121, 167),(0, 0, 0),     (255, 255, 255),
]

# ─── State global ──────────────────────────────────────────────────────────
app              = Flask(__name__)
model            = None
latest_frame     = None      # Frame terbaru dengan anotasi (untuk streaming)
frame_lock       = threading.Lock()
stats = {
    "total_frames"    : 0,
    "total_detections": 0,
    "avg_inference_ms": 0.0,
    "fps"             : 0.0,
}
fps_history = deque(maxlen=30)
last_frame_time = time.time()


# ══════════════════════════════════════════════════════════════════════════
def load_model():
    """Muat model YOLOv8n ke memori."""
    global model
    print(f"[Model] Memuat {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Warmup: satu inferensi dengan gambar dummy agar latensi pertama lebih rendah
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    print("[Model] YOLOv8n siap.")


def draw_detections(frame: np.ndarray, results) -> tuple[np.ndarray, list]:
    """
    Gambar bounding box dan label pada frame,
    kembalikan frame yang sudah dianotasi beserta list deteksi.
    """
    detections = []
    names = model.names   # Dict {id: nama_kelas}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Koordinat bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = names[cls]

            # Pilih warna berdasarkan kelas
            color = COLORS[cls % len(COLORS)]

            # Gambar kotak
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Background label
            label_text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Teks label
            cv2.putText(
                frame, label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )

            detections.append({
                "label"     : label,
                "confidence": round(conf, 4),
                "class_id"  : cls,
                "bbox"      : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    return frame, detections


def update_stats(inference_ms: float, n_detections: int):
    """Perbarui statistik performa server."""
    global last_frame_time

    stats["total_frames"]     += 1
    stats["total_detections"] += n_detections

    # FPS berbasis sliding window
    now = time.time()
    fps_history.append(now - last_frame_time)
    last_frame_time = now
    avg_interval = sum(fps_history) / len(fps_history)
    stats["fps"] = round(1.0 / avg_interval if avg_interval > 0 else 0.0, 1)

    # Moving average inference time
    alpha = 0.1
    stats["avg_inference_ms"] = round(
        alpha * inference_ms + (1 - alpha) * stats["avg_inference_ms"], 1
    )


# ══════════════════════════════════════════════════════════════════════════
@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Menerima frame JPEG (multipart atau raw binary),
    menjalankan YOLOv8n, dan mengembalikan JSON.
    """
    global latest_frame

    # ── Ambil data gambar dari request ──────────────────────────────────
    if "frame" in request.files:
        # Multipart form-data (dikirim oleh sketch Arduino)
        file_data = request.files["frame"].read()
    elif request.data:
        # Raw binary (fallback)
        file_data = request.data
    else:
        return jsonify({"error": "Tidak ada frame diterima"}), 400

    # ── Decode JPEG → numpy array ────────────────────────────────────────
    nparr = np.frombuffer(file_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Gagal mendekode gambar"}), 422

    # ── Inferensi YOLOv8n ────────────────────────────────────────────────
    t_start = time.perf_counter()

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRES,
        iou=IOU_THRES,
        verbose=False,
    )

    inference_ms = (time.perf_counter() - t_start) * 1000

    # ── Anotasi frame ────────────────────────────────────────────────────
    annotated, detections = draw_detections(frame.copy(), results)

    # Tambahkan overlay info (FPS + inference time)
    overlay = f"FPS: {stats['fps']} | Inferensi: {inference_ms:.1f}ms"
    cv2.putText(
        annotated, overlay, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )

    # Simpan frame terbaru untuk streaming
    with frame_lock:
        latest_frame = annotated.copy()

    # ── Perbarui statistik ───────────────────────────────────────────────
    update_stats(inference_ms, len(detections))

    # ── Kembalikan JSON ──────────────────────────────────────────────────
    return jsonify({
        "status"      : "ok",
        "count"       : len(detections),
        "detections"  : detections,
        "inference_ms": round(inference_ms, 2),
        "fps"         : stats["fps"],
    })


@app.route("/stream")
def stream():
    """
    GET /stream
    MJPEG stream dengan bounding box untuk ditampilkan di browser.
    Buka di: http://<IP_SERVER>:5000/stream
    """
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame

            if frame is None:
                # Tampilkan placeholder jika belum ada frame
                placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "Menunggu ESP32-CAM...",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (100, 100, 100), 1
                )
                frame = placeholder

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(0.05)  # Batasi stream ke ~20 FPS

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """GET /status — Kembalikan statistik server."""
    return jsonify({
        "status"          : "running",
        "model"           : MODEL_PATH,
        "confidence_thres": CONFIDENCE_THRES,
        "img_size"        : IMG_SIZE,
        **stats,
    })


@app.route("/")
def index():
    """GET / — Halaman dashboard sederhana."""
    html = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>ESP32-CAM YOLOv8n Dashboard</title>
        <style>
            body { font-family: monospace; background:#111; color:#eee; margin:20px; }
            h1   { color:#4fc3f7; }
            img  { border:2px solid #4fc3f7; border-radius:4px; max-width:100%; }
            .stats { background:#1e1e1e; padding:12px; border-radius:6px; margin-top:16px; }
            .stat  { display:flex; justify-content:space-between; padding:4px 0;
                     border-bottom:1px solid #333; }
        </style>
        <script>
            async function refreshStats() {
                const r   = await fetch('/status');
                const d   = await r.json();
                document.getElementById('fps').textContent        = d.fps + ' FPS';
                document.getElementById('inf').textContent        = d.avg_inference_ms + ' ms';
                document.getElementById('frames').textContent    = d.total_frames;
                document.getElementById('detected').textContent  = d.total_detections;
            }
            setInterval(refreshStats, 1000);
            refreshStats();
        </script>
    </head>
    <body>
        <h1>ESP32-CAM + YOLOv8n</h1>
        <img src="/stream" alt="Live Detection Stream">
        <div class="stats">
            <div class="stat"><span>FPS</span>           <span id="fps">–</span></div>
            <div class="stat"><span>Inferensi</span>     <span id="inf">–</span></div>
            <div class="stat"><span>Total Frame</span>   <span id="frames">–</span></div>
            <div class="stat"><span>Total Objek</span>   <span id="detected">–</span></div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_model()

    print(f"\n[Server] Berjalan di http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"[Server] Dashboard : http://localhost:{SERVER_PORT}/")
    print(f"[Server] Stream    : http://localhost:{SERVER_PORT}/stream")
    print(f"[Server] Status    : http://localhost:{SERVER_PORT}/status\n")

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        threaded=True,
    )