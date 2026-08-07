import os
from ultralytics import YOLO
import cv2

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "best.pt")

# Load the model once at import time to keep response time lower.
try:
    model = YOLO(MODEL_PATH)
except Exception as load_error:
    model = None
    load_exception = load_error


def detect_ringworm(image_path):
    """Run YOLO detection on an image and return structured results."""
    if model is None:
        raise RuntimeError(f"Unable to load model: {load_exception}")

    # Perform inference.
    results = model(image_path, imgsz=640, conf=0.25, max_det=10)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return {
            "detected": False,
            "confidence": 0.0,
            "result_image": save_default_image(image_path),
            "label": "No Ringworm Detected",
        }

    best_box = max(results[0].boxes, key=lambda box: box.conf)
    confidence = float(best_box.conf)
    label = "Ringworm"

    annotated_image_path = save_annotated_image(image_path, results[0], confidence, label)

    return {
        "detected": True,
        "confidence": round(confidence, 2),
        "result_image": os.path.basename(annotated_image_path),
        "label": label,
    }


def save_default_image(image_path):
    """Copy the original image to the results folder when no detection occurs."""
    target_name = os.path.basename(image_path)
    target_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "results", target_name)
    image = cv2.imread(image_path)
    cv2.imwrite(target_path, image)
    return target_path


def save_annotated_image(image_path, result, confidence, label):
    """Draw bounding boxes and labels on the image and save it."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Unable to open uploaded image.")

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        text = f"{label} {int(box.conf[0] * 100)}%"
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            image,
            text,
            (x1, y1 - 10 if y1 - 10 > 20 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    target_name = f"result_{os.path.basename(image_path)}"
    target_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "results", target_name)
    cv2.imwrite(target_path, image)
    return target_path
