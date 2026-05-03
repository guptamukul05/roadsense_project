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
import shutil
import subprocess
import time
import uuid
import json
import base64
import threading
import numpy as np
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from ultralytics import YOLO
import supervision as sv

from location_utils import (
    enrich_location_strings,
    extract_gps_from_image_path,
    extract_gps_from_video_path,
    extract_gps_from_upload,
    reverse_geocode,
)

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
    else:
        print(f"Model 1 not found at {MODEL1_PATH}")
except Exception as e:
    print(f"Model 1 failed to load: {e}")

try:
    if MODEL2_PATH.is_file():
        model2 = YOLO(str(MODEL2_PATH))
        names2 = model2.names
    else:
        print(f"Model 2 not found at {MODEL2_PATH}")
except Exception as e:
    print(f"Model 2 failed to load: {e}")

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


def resolve_geo_fields(
    lat_in: str,
    lng_in: str,
    media_path: Optional[str] = None,
    media_kind: Optional[str] = None,
) -> tuple[Optional[float], Optional[float], str, str]:
    """
    Fill lat/lng from EXIF or video metadata when form fields are empty;
    reverse-geocode to city, state.
    Returns lat/lng as floats when coordinates exist, else None.
    """
    lat = (lat_in or "").strip()
    lng = (lng_in or "").strip()
    if (not lat or not lng) and media_path and media_kind == "image":
        elat, elng, _ = extract_gps_from_image_path(media_path)
        if elat is not None and elng is not None:
            lat, lng = str(elat), str(elng)
    elif (not lat or not lng) and media_path and media_kind == "video":
        elat, elng, _ = extract_gps_from_video_path(media_path)
        if elat is not None and elng is not None:
            lat, lng = str(elat), str(elng)
    city, state, _ = enrich_location_strings(lat, lng)
    lat_f: Optional[float] = None
    lng_f: Optional[float] = None
    if lat and lng:
        try:
            lat_f = round(float(lat), 6)
            lng_f = round(float(lng), 6)
        except ValueError:
            pass
    return lat_f, lng_f, city, state


def merge_manual_city_state(
    city: str,
    state: str,
    city_form: str,
    state_form: str,
) -> tuple[str, str]:
    """Non-empty `city` / `state` form fields override reverse-geocoded values."""
    c = (city_form or "").strip()
    s = (state_form or "").strip()
    if c:
        city = c
    if s:
        state = s
    return city, state


def transcode_mp4_h264_web(src_path: str, dst_path: str) -> bool:
    """H.264 + yuv420p + faststart — HTML5 `<video>` compatible (OpenCV mp4v often is not)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", src_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                dst_path,
            ],
            check=True,
            timeout=3600,
        )
        return os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False

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


@app.route("/api/geocode/reverse", methods=["GET", "POST"])
def api_geocode_reverse():
    """Return Nominatim city / state for lat, lon (server-side only)."""
    lat = lon = None
    if request.method == "POST":
        data = request.get_json(silent=True)
        if data:
            lat = data.get("lat")
            lon = data.get("lon", data.get("lng"))
        if lat is None and request.form:
            lat = request.form.get("lat")
            lon = request.form.get("lon", request.form.get("lng"))
    if lat is None:
        lat = request.args.get("lat")
        lon = request.args.get("lon", request.args.get("lng"))
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid lat or lon"}), 400
    geo = reverse_geocode(lat_f, lon_f)
    return jsonify({
        "city": geo.get("city") or "",
        "state": geo.get("state") or "",
    })


@app.route("/detect/location", methods=["POST"])
def detect_location():
    """Extract GPS from image EXIF or video metadata, then reverse-geocode via Nominatim."""
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGES and ext not in ALLOWED_VIDEOS:
        return jsonify({"error": "Invalid file format"}), 400

    lat, lng, src, err = extract_gps_from_upload(file, ext, ALLOWED_IMAGES, ALLOWED_VIDEOS)
    if lat is None or lng is None:
        return jsonify({"error": err or "No GPS metadata found"}), 400

    geo = reverse_geocode(lat, lng)
    return jsonify({
        "lat": round(float(lat), 6),
        "lng": round(float(lng), 6),
        "city": geo.get("city") or "",
        "state": geo.get("state") or "",
        "source_gps": src,
        "reverse_geocode_error": geo.get("error"),
    })


# ── IMAGE UPLOAD ──
@app.route("/detect/image", methods=["POST"])
def detect_image():
    file    = request.files.get("file")
    use_m1  = request.form.get("use_m1") == "true"
    use_m2  = request.form.get("use_m2") == "true"
    conf1   = float(request.form.get("conf1", DEFAULT_CONF1))
    conf2   = float(request.form.get("conf2", DEFAULT_CONF2))
    lat        = request.form.get("lat", "")
    lng        = request.form.get("lng", "")
    city_form  = request.form.get("city", "").strip()
    state_form = request.form.get("state", "").strip()

    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGES:
        return jsonify({"error": "Invalid image format"}), 400

    fname    = f"{uuid.uuid4().hex}.{ext}"
    fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    lat, lng, city, state = resolve_geo_fields(lat, lng, fpath, "image")
    city, state = merge_manual_city_state(city, state, city_form, state_form)

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
        "lat": lat, "lng": lng,
        "city":  city,
        "state": state,
    }
    save_history(entry)

    return jsonify({
        "original_url": url_for("static", filename=f"uploads/{fname}"),
        "result_url":   url_for("static", filename=f"results/{rname}"),
        "detections":   detections,
        "score":        score,
        "severity":     severity_label(score),
        "entry_id":     entry["id"],
        "lat":          lat,
        "lng":          lng,
        "city":         city,
        "state":        state,
    })

# ── VIDEO UPLOAD ──
@app.route("/detect/video", methods=["POST"])
def detect_video():
    file   = request.files.get("file")
    use_m1 = request.form.get("use_m1") == "true"
    use_m2 = request.form.get("use_m2") == "true"
    conf1  = float(request.form.get("conf1", DEFAULT_CONF1))
    conf2  = float(request.form.get("conf2", DEFAULT_CONF2))
    lat        = request.form.get("lat", "")
    lng        = request.form.get("lng", "")
    city_form  = request.form.get("city", "").strip()
    state_form = request.form.get("state", "").strip()

    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_VIDEOS:
        return jsonify({"error": "Invalid video format"}), 400

    input_tmp_path = None
    output_tmp_path = None
    cap = None
    writer = None
    
    try:
        # ─── SAVE INPUT TO TEMP FILE ───
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            input_tmp_path = tmp.name
            tmp.write(file.read())

        lat, lng, city, state = resolve_geo_fields(lat, lng, input_tmp_path, "video")
        city, state = merge_manual_city_state(city, state, city_form, state_form)

        # ─── OPEN INPUT VIDEO ───
        cap = cv2.VideoCapture(input_tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Cannot read input video"}), 400
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if raw_fps <= 0.01 or raw_fps != raw_fps:  # missing, zero, or NaN
            raw_fps = 30.0
        fps = max(1.0, min(raw_fps, 120.0))

        if width <= 1 or height <= 1:
            return jsonify({"error": "Cannot read video dimensions"}), 400
        
        # Ensure even dimensions (required by many codecs)
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        # ─── CREATE OUTPUT TEMP FILE ───
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            output_tmp_path = tmp.name
        
        # ─── INITIALIZE VIDEO WRITER ───
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_tmp_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            return jsonify({"error": "VideoWriter initialization failed"}), 500

        # ─── PROCESS VIDEO ───
        all_dets, all_scores = [], []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame is None or frame.size == 0:
                frame_idx += 1
                continue
            
            # Resize if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            ann, dets, score = process_frame(frame, use_m1, use_m2, conf1, conf2)
            all_dets.extend(dets)
            all_scores.append(score)
            
            # Ensure proper format
            if ann is None or ann.size == 0:
                ann = frame
            if ann.dtype != np.uint8:
                ann = np.uint8(np.clip(ann, 0, 255))
            if ann.shape != (height, width, 3):
                ann = cv2.resize(ann, (width, height))
            
            writer.write(ann)
            
            frame_idx += 1
        
        # ─── FINALIZE (release once; finally must not double-release or MP4 moov is corrupted)
        if cap is not None:
            cap.release()
            cap = None
        if writer is not None:
            writer.release()
            writer = None
        
        # OpenCV's mp4v is MPEG-4 Part 2 — often won't play in Safari / some browsers
        if os.path.exists(output_tmp_path) and os.path.getsize(output_tmp_path) > 0:
            fd, web_tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            if transcode_mp4_h264_web(output_tmp_path, web_tmp):
                try:
                    os.remove(output_tmp_path)
                except OSError:
                    pass
                output_tmp_path = web_tmp
            else:
                try:
                    os.remove(web_tmp)
                except OSError:
                    pass
        
        if os.path.exists(output_tmp_path):
            # Move to final location
            rname = f"result_{uuid.uuid4().hex}.mp4"
            rpath = os.path.join(app.config["RESULT_FOLDER"], rname)
            os.rename(output_tmp_path, rpath)
        else:
            return jsonify({"error": "Output file creation failed"}), 500
        
        # ─── SAVE HISTORY ───
        avg_score = round(sum(all_scores)/len(all_scores)) if all_scores else 1
        entry = {
            "id":         uuid.uuid4().hex[:8],
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type":       "video",
            "original":   file.filename,
            "result":     rname,
            "detections": all_dets[:20],
            "score":      avg_score,
            "severity":   severity_label(avg_score),
            "lat": lat, "lng": lng,
            "city":  city,
            "state": state,
        }
        save_history(entry)
        
        return jsonify({
            "result_url": url_for("static", filename=f"results/{rname}"),
            "detections": all_dets[:20],
            "score":      avg_score,
            "severity":   severity_label(avg_score),
            "entry_id":   entry["id"],
            "lat":        lat,
            "lng":        lng,
            "city":       city,
            "state":      state,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Video processing error: {str(e)}"}), 500
        
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        
        if input_tmp_path and os.path.exists(input_tmp_path):
            try:
                os.remove(input_tmp_path)
            except OSError:
                pass
        
        if output_tmp_path and os.path.exists(output_tmp_path):
            try:
                os.remove(output_tmp_path)
            except OSError:
                pass

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
