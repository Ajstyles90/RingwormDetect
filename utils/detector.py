import os
import cv2
import torch
from ultralytics import YOLO

# Keep PyTorch from creating too many CPU threads on Render
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model once
try:
    model = YOLO(MODEL_PATH)
    model.to("cpu")
    load_exception = None
except Exception as error:
    model = None
    load_exception = error


def detect_ringworm(image_path):
    """Run lightweight YOLO detection on CPU."""

    if model is None:
        raise RuntimeError(
            f"Unable to load model: {load_exception}"
        )

    # Smaller image size = significantly lower memory usage
    results = model.predict(
        source=image_path,
        imgsz=320,
        conf=0.25,
        max_det=1,
        device="cpu",
        verbose=False,
        half=False
    )

    if not results or len(results[0].boxes) == 0:
        return {
            "detected": False,
            "confidence": 0.0,
            "result_image": os.path.basename(
                save_default_image(image_path)
            ),
            "label": "No Ringworm Detected",
        }

    best_box = max(
        results[0].boxes,
        key=lambda box: float(box.conf[0])
    )

    confidence = float(best_box.conf[0])
    label = "Ringworm"

    annotated_image_path = save_annotated_image(
        image_path,
        results[0],
        confidence,
        label
    )

    return {
        "detected": True,
        "confidence": round(confidence, 2),
        "result_image": os.path.basename(
            annotated_image_path
        ),
        "label": label,
    }


def save_default_image(image_path):
    """Copy original image to results folder."""

    target_name = os.path.basename(image_path)
    target_path = os.path.join(
        RESULTS_DIR,
        target_name
    )

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            "Unable to open uploaded image."
        )

    cv2.imwrite(target_path, image)

    return target_path


def save_annotated_image(
    image_path,
    result,
    confidence,
    label
):
    """Draw detection boxes and save result."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            "Unable to open uploaded image."
        )

    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        text = f"{label} {int(confidence * 100)}%"

        color = (255, 0, 0)
        thickness = 2

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        cv2.putText(
            image,
            text,
            (
                x1,
                y1 - 10 if y1 - 10 > 20 else y1 + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    target_name = (
        f"result_{os.path.basename(image_path)}"
    )

    target_path = os.path.join(
        RESULTS_DIR,
        target_name
    )

    cv2.imwrite(target_path, image)

    return target_path