# RoadSense AI — Setup Guide

## Your Final Folder Structure

```
backend/                          ← your backend folder
├── app.py                        ← Flask backend (this file)
├── requirements.txt
├── templates/
│   ├── index.html                ← Dashboard 1 (Upload)
│   ├── analytics.html            ← Dashboard 2 (Analytics + Map)
│   └── history_page.html         ← History
├── static/
│   ├── uploads/                  ← auto-created
│   └── results/                  ← auto-created
├── RoadDetectionModel/           ← copy from your friend's repo
│   └── RoadModel_yolov8m.pt_rounds120_b9/
│       └── weights/
│           └── best.pt
└── YOLOv8_Small_2nd_Model.pt     ← copy from your friend's repo
```

---

## Step 1 — Copy model files

From your friend's repo, copy these into your `backend/` folder:
- The entire `RoadDetectionModel/` folder
- `YOLOv8_Small_2nd_Model.pt`

---

## Step 2 — Install dependencies

Open terminal inside your `backend/` folder:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run the app

```bash
python app.py
```

Open browser → http://localhost:5000

---

## What each page does

| URL | Page |
|-----|------|
| http://localhost:5000/ | Upload Dashboard — image, video, live camera |
| http://localhost:5000/analytics | Analytics — GPS map, table, municipality alerts |
| http://localhost:5000/history_page | History — all past detections with images |

---

## Features

- ✅ Image upload with detection
- ✅ Video upload with detection  
- ✅ Live camera stream with detection
- ✅ Model 1 (red boxes) + Model 2 (blue boxes) — individually toggleable
- ✅ Confidence threshold slider for each model
- ✅ Severity score (1–10) with color-coded display
- ✅ GPS location (auto from EXIF or manual input)
- ✅ Real Leaflet.js map with GPS pins
- ✅ Detection history stored in history.json
- ✅ Municipality notification system
- ✅ Analytics dashboard with live map + table

---

## API Endpoints (for your backend friend)

| Method | URL | What it does |
|--------|-----|--------------|
| POST | /detect/image | Run detection on image |
| POST | /detect/video | Run detection on video |
| POST | /camera/start | Start live camera |
| POST | /camera/stop | Stop live camera |
| GET | /camera/stream | MJPEG camera stream |
| GET | /history | Get all history as JSON |
| POST | /history/clear | Clear history |
| GET | /status | Model + camera status |
