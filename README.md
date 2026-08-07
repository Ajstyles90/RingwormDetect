# Ringworm Detection System

A simple Python + Flask application for detecting ringworm in skin images using a trained YOLOv11 model.

## Project Overview

This beginner-friendly app allows you to upload a skin image, run ringworm detection, and display the results with bounding boxes and confidence scores.

## Features

- Upload a skin image via form or drag-and-drop
- Preview image before detection
- Run YOLOv11 inference using `model/best.pt`
- Display original and detected images
- Show detection status and confidence score
- Save result image with annotated bounding boxes
- Download detected image

## Installation

1. Clone or download the project files.
2. Copy your `best.pt` model into the `model/` folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Replace the Model

To use a different YOLOv11 model, replace `model/best.pt` with your own model file. Keep the filename `best.pt` or update `MODEL_PATH` in `utils/detector.py`.

## How Detection Works

- The backend receives the uploaded image in `/predict`.
- `utils/detector.py` loads the YOLOv11 model.
- The model runs inference and returns bounding boxes and confidence values.
- Detected boxes and labels are drawn onto the saved result image.
- The UI shows both original and result images, plus detection details.

## Folder Structure

```
ringworm-system/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   └── best.pt
├── static/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── scripts.js
│   ├── uploads/
│   └── results/
├── templates/
│   └── index.html
└── utils/
    └── detector.py
```

## Notes

- Do not retrain the model.
- Only use the provided uploaded YOLOv11 model.
- Keep uploads under 5 MB.
