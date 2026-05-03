"""
RoadSense AI — Flask Backend
Integrates Model 1 (best.pt) + Model 2 (YOLOv8_Small_2nd_Model.pt)
Supports: image upload, video upload, live camera stream, history, severity scoring
"""

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, Response, send_from_directory
)
import cv2
import os
import time
import uuid
import json
import base64
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

from ultralytics import YOLO
import supervision as sv

# ─── APP CONFIG ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "roadsense_ai_secret_2024"
app.config["UPLOAD_FOLDER"]  = "static/uploads"
app.config["RESULT_FOLDER"]  = "static/results"
app.config["HISTORY_FILE"]   = "static/history.json"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv"}

for folder in [app.config["UPLOAD_FOLDER"], app.config["RESULT_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# ─── MODEL PATHS  (adjust if needed) ─────────────────────────
BASE = Path(__file__).parent
#MODEL1_PATH = BASE / "../RoadDetectionModel/RoadModel_yolov8m.pt_rounds120_b9/weights/best.pt"
#MODEL2_PATH = BASE / "../YOLOv8_Small_2nd_Model.pt"


#BASE = Path(__file__).parent

MODEL1_PATH = BASE / "RoadDetectionModel" / "RoadModel_yolov8m.pt_rounds120_b9" / "weights" / "best.pt"
MODEL2_PATH = BASE / "YOLOv8_Small_2nd_Model.pt"


# MODEL1_PATH = YOLO("C:\\Users\\raju9\\OneDrive\\Desktop\\roadsense\\roadsense_complete\\roadsense\\RoadDetectionModel")
# MODEL2_PATH = YOLO("C:\\Users\\raju9\\OneDrive\\Desktop\\roadsense\\roadsense_complete\\roadsense\\YOLOv8_Small_2nd_Model.pt")

DEFAULT_CONF1 = 0.35
DEFAULT_CONF2 = 0.40

# ─── LOAD MODELS ─────────────────────────────────────────────
model1 = model2 = None
names1 = names2 = {}

try:
    if MODEL1_PATH.is_file():
        model1 = YOLO(str(MODEL1_PATH))
        names1 = model1.names
        print("✅ Model 1 loaded")
    else:
        print(f"⚠️  Model 1 not found at {MODEL1_PATH}")
except Exception as e:
    print(f"❌ Model 1 failed: {e}")

try:
    if MODEL2_PATH.is_file():
        model2 = YOLO(str(MODEL2_PATH))
        names2 = model2.names
        print("✅ Model 2 loaded")
    else:
        print(f"⚠️  Model 2 not found at {MODEL2_PATH}")
except Exception as e:
    print(f"❌ Model 2 failed: {e}")

# ─── SUPERVISION ANNOTATORS ───────────────────────────────────
box1   = sv.BoxAnnotator(thickness=2, color=sv.Color.RED)
label1 = sv.LabelAnnotator(text_thickness=1, text_scale=0.55,
                            text_color=sv.Color.BLACK, text_padding=2,
                            text_position=sv.Position.TOP_LEFT)
box2   = sv.BoxAnnotator(thickness=2, color=sv.Color.BLUE)
label2 = sv.LabelAnnotator(text_thickness=1, text_scale=0.55,
                            text_color=sv.Color.BLACK, text_padding=2,
                            text_position=sv.Position.TOP_RIGHT)

# ─── SEVERITY SCORING (from inference.py) ────────────────────
def compute_severity(detections_list, names_map, img_area):
    raw = 0.0
    for detections, names in zip(detections_list, names_map):
        for i in range(len(detections)):
            cls  = names.get(int(detections.class_id[i]), "").lower()
            x1,y1,x2,y2 = detections.xyxy[i]
            ratio = ((x2-x1)*(y2-y1)) / max(img_area, 1)
            weight = 8.0 if "pothole" in cls else 6.0 if "crack-severe" in cls else 2.0 if "crack" in cls else 1.0
            raw += weight * (1 + ratio * 10)
    if raw == 0:
        return 1
    scaled = (raw / 30.0) * 9 + 1
    return min(10, max(1, round(scaled)))

def severity_label(score):
    if score >= 9: return "Critical"
    if score >= 7: return "High"
    if score >= 4: return "Medium"
    return "Low"

# ─── CORE FRAME PROCESSOR ─────────────────────────────────────
def process_frame(frame, use_m1=True, use_m2=True, conf1=DEFAULT_CONF1, conf2=DEFAULT_CONF2):
    """Run enabled models on a frame, return annotated frame + detection info."""
    annotated = frame.copy()
    img_area  = frame.shape[0] * frame.shape[1]
    detections_list, names_list, all_dets = [], [], []

    if use_m1 and model1:
        res = model1.predict(frame, conf=conf1, verbose=False)[0]
        d   = sv.Detections.from_ultralytics(res)
        detections_list.append(d); names_list.append(names1)
        lbs = [f"M1:{names1.get(int(c),str(c))} {v:.2f}" for c,v in zip(d.class_id, d.confidence)]
        annotated = box1.annotate(annotated, d)
        annotated = label1.annotate(annotated, d, labels=lbs)
        for c,v in zip(d.class_id, d.confidence):
            all_dets.append({"model":"Model 1","class":names1.get(int(c),str(c)),"confidence":round(float(v),2)})

    if use_m2 and model2:
        res = model2.predict(frame, conf=conf2, verbose=False)[0]
        d   = sv.Detections.from_ultralytics(res)
        detections_list.append(d); names_list.append(names2)
        lbs = [f"M2:{names2.get(int(c),str(c))} {v:.2f}" for c,v in zip(d.class_id, d.confidence)]
        annotated = box2.annotate(annotated, d)
        annotated = label2.annotate(annotated, d, labels=lbs)
        for c,v in zip(d.class_id, d.confidence):
            all_dets.append({"model":"Model 2","class":names2.get(int(c),str(c)),"confidence":round(float(v),2)})

    score = compute_severity(detections_list, names_list, img_area)
    return annotated, all_dets, score

# ─── HISTORY HELPERS ──────────────────────────────────────────
def load_history():
    hf = app.config["HISTORY_FILE"]
    if os.path.exists(hf):
        try:
            with open(hf) as f:
                return json.load(f)
        except:
            pass
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]           # keep last 50
    with open(app.config["HISTORY_FILE"], "w") as f:
        json.dump(history, f)

def frame_to_b64(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()

# ─── LIVE CAMERA STATE ────────────────────────────────────────
camera_state = {
    "running": False,
    "frame":   None,
    "lock":    threading.Lock(),
    "thread":  None,
    "conf1":   DEFAULT_CONF1,
    "conf2":   DEFAULT_CONF2,
    "use_m1":  True,
    "use_m2":  True,
}

def camera_worker():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_state["running"] = False
        return
    while camera_state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        try:
            ann, _, _ = process_frame(
                frame,
                use_m1=camera_state["use_m1"],
                use_m2=camera_state["use_m2"],
                conf1=camera_state["conf1"],
                conf2=camera_state["conf2"],
            )
        except:
            ann = frame
        with camera_state["lock"]:
            camera_state["frame"] = ann
    cap.release()

def gen_camera_stream():
    while camera_state["running"]:
        with camera_state["lock"]:
            frame = camera_state["frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.04)

# ─── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           model1_available=model1 is not None,
                           model2_available=model2 is not None)

# ── IMAGE UPLOAD ──
@app.route("/detect/image", methods=["POST"])
def detect_image():
    file    = request.files.get("file")
    use_m1  = request.form.get("use_m1") == "true"
    use_m2  = request.form.get("use_m2") == "true"
    conf1   = float(request.form.get("conf1", DEFAULT_CONF1))
    conf2   = float(request.form.get("conf2", DEFAULT_CONF2))
    lat     = request.form.get("lat", "")
    lng     = request.form.get("lng", "")
    loc     = request.form.get("loc", "")

    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGES:
        return jsonify({"error": "Invalid image format"}), 400

    fname    = f"{uuid.uuid4().hex}.{ext}"
    fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    img = cv2.imread(fpath)
    if img is None:
        return jsonify({"error": "Cannot read image"}), 400

    annotated, detections, score = process_frame(img, use_m1, use_m2, conf1, conf2)

    rname = f"result_{uuid.uuid4().hex}.jpg"
    rpath = os.path.join(app.config["RESULT_FOLDER"], rname)
    cv2.imwrite(rpath, annotated)

    # Save history entry
    entry = {
        "id":         uuid.uuid4().hex[:8],
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "image",
        "original":   fname,
        "result":     rname,
        "detections": detections,
        "score":      score,
        "severity":   severity_label(score),
        "lat": lat, "lng": lng, "loc": loc,
    }
    save_history(entry)

    return jsonify({
        "original_url": url_for("static", filename=f"uploads/{fname}"),
        "result_url":   url_for("static", filename=f"results/{rname}"),
        "detections":   detections,
        "score":        score,
        "severity":     severity_label(score),
        "entry_id":     entry["id"],
    })

# ── VIDEO UPLOAD ──
@app.route("/detect/video", methods=["POST"])
def detect_video():
    file   = request.files.get("file")
    use_m1 = request.form.get("use_m1") == "true"
    use_m2 = request.form.get("use_m2") == "true"
    conf1  = float(request.form.get("conf1", DEFAULT_CONF1))
    conf2  = float(request.form.get("conf2", DEFAULT_CONF2))
    lat    = request.form.get("lat", "")
    lng    = request.form.get("lng", "")
    loc    = request.form.get("loc", "")

    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_VIDEOS:
        return jsonify({"error": "Invalid video format"}), 400

    fname = f"{uuid.uuid4().hex}.{ext}"
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    rname = f"result_{uuid.uuid4().hex}.mp4"
    rpath = os.path.join(app.config["RESULT_FOLDER"], rname)

    cap = cv2.VideoCapture(fpath)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(rpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_dets, all_scores = [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % 2 == 0:    # process every 2nd frame for speed
            ann, dets, score = process_frame(frame, use_m1, use_m2, conf1, conf2)
            all_dets.extend(dets)
            all_scores.append(score)
            last_ann = ann
        out.write(last_ann if frame_idx % 2 == 1 else ann)
        frame_idx += 1

    cap.release(); out.release()

    avg_score = round(sum(all_scores)/len(all_scores)) if all_scores else 1

    entry = {
        "id":         uuid.uuid4().hex[:8],
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "video",
        "original":   fname,
        "result":     rname,
        "detections": all_dets[:20],
        "score":      avg_score,
        "severity":   severity_label(avg_score),
        "lat": lat, "lng": lng, "loc": loc,
    }
    save_history(entry)

    return jsonify({
        "result_url": url_for("static", filename=f"results/{rname}"),
        "detections": all_dets[:20],
        "score":      avg_score,
        "severity":   severity_label(avg_score),
        "entry_id":   entry["id"],
    })

# ── LIVE CAMERA ──
@app.route("/camera/start", methods=["POST"])
def camera_start():
    data = request.get_json(silent=True) or {}
    camera_state["conf1"]  = float(data.get("conf1", DEFAULT_CONF1))
    camera_state["conf2"]  = float(data.get("conf2", DEFAULT_CONF2))
    camera_state["use_m1"] = data.get("use_m1", True)
    camera_state["use_m2"] = data.get("use_m2", True)
    if not camera_state["running"]:
        camera_state["running"] = True
        t = threading.Thread(target=camera_worker, daemon=True)
        camera_state["thread"] = t
        t.start()
    return jsonify({"status": "started"})

@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    camera_state["running"] = False
    return jsonify({"status": "stopped"})

@app.route("/camera/stream")
def camera_stream():
    return Response(gen_camera_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── HISTORY ──
@app.route("/history")
def history():
    return jsonify(load_history())

@app.route("/history/clear", methods=["POST"])
def history_clear():
    hf = app.config["HISTORY_FILE"]
    if os.path.exists(hf):
        os.remove(hf)
    return jsonify({"status": "cleared"})

# ── STATUS ──
@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/history_page")
def history_page():
    return render_template("history_page.html")

@app.route("/status")
def status():
    return jsonify({
        "model1": model1 is not None,
        "model2": model2 is not None,
        "camera": camera_state["running"],
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
