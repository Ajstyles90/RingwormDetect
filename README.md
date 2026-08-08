# Ringworm Detection Using The YOLOv11s Architecture

An AI-assisted ringworm detection web application developed as a Computer Science project. It uses a trained **YOLOv11s** model (exported to ONNX) to analyze skin images for visual indications of ringworm.

**Made by Al John Aguilar** · Catanduanes State University

> This is an AI-assisted research and detection tool. Results are not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.

## Features

- Modern, responsive, black/white/neutral interface (desktop, tablet, mobile)
- Home, Detection, and About pages with clean navigation
- Drag & drop or click-to-upload image interface
- Image preview with replace / remove controls
- Real YOLOv11s inference via ONNX Runtime (optimized CPU session)
- Detection status, confidence score, and detection count
- Bounding-box overlay drawn on the original image
- Annotated result image with download option
- Professional loading animation and friendly error messages
- Web deployment ready for Render (gunicorn)

## How It Works

1. User uploads an image on the Detection page
2. The image is sent to the backend at `POST /api/predict`
3. YOLOv11s analyzes the image (320×320 input, single class)
4. Detection results are returned as JSON (status, confidence, boxes, result image)
5. Results are presented visually with bounding boxes

## Project Structure

```
ringworm-system/
├── app.py                 # Flask app + routes + JSON API
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── .env.example           # Example environment variables
├── model/
│   └── best.onnx          # Trained YOLOv11s model (ONNX export)
├── static/
│   ├── css/styles.css
│   ├── js/scripts.js
│   ├── uploads/           # Uploaded images (gitignored)
│   └── results/           # Result images (gitignored)
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── detection.html
│   └── about.html
└── utils/
    └── detector.py        # ONNX inference + post-processing
```

## Requirements

- Python 3.10+ (recommended 3.11)
- `pip install -r requirements.txt`

Dependencies: Flask, gunicorn, opencv-python-headless, numpy, onnxruntime.

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open in your browser
# http://127.0.0.1:5000
```

## Deployment on Render

This repository includes `render.yaml` for one-click Render deployment.

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from the repository (Render will detect `render.yaml`).
3. Render builds and starts the service automatically.
4. The model (`model/best.onnx`) is committed and loaded at startup.

### Manual Render setup

- **Runtime**: Python
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
- **Environment variable**: set a strong `SECRET_KEY`

Note: Render free instances sleep after inactivity; the ONNX model loads on first request, so the first analysis after waking may take a few extra seconds.

## API

### `POST /api/predict`

Accepts a multipart form field `image` (PNG, JPG, JPEG, WEBP, GIF; max 5 MB).

Successful response:

```json
{
  "success": true,
  "detected": true,
  "confidence": 0.61,
  "detection_count": 3,
  "boxes": [
    { "x1": 120, "y1": 80, "x2": 260, "y2": 240, "confidence": 0.61 }
  ],
  "label": "Ringworm Detected",
  "result_image": "result_ringworm3.jpg",
  "result_image_url": "/static/results/result_ringworm3.jpg",
  "time_seconds": 0.42
}
```

## Replace the Model

To use a different trained YOLOv11 model, replace `model/best.onnx` with your own ONNX export and keep the filename or update `MODEL_PATH` in `utils/detector.py`.

## Notes

- The model's ONNX export is used for inference (input `[1,3,320,320]`, output `[1,5,2100]`).
- Uploaded images and generated results are stored under `static/uploads/` and `static/results/` and are not committed to git.

