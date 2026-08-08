import os
import cv2
import gc
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best.onnx"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

model = None


def get_model():
    global model

    if model is None:
        print(f"Loading ONNX model from: {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        model = YOLO(MODEL_PATH, task="detect")

        print("ONNX model loaded successfully.")

    return model


def detect_ringworm(image_path):
    """Run YOLO ONNX detection on an image."""

    detector = get_model()

    try:
        print(f"Processing image: {image_path}")

        results = detector.predict(
            source=image_path,
            imgsz=320,
            conf=0.25,
            max_det=5,
            device="cpu",
            verbose=False
        )

        if not results:
            raise RuntimeError(
                "No prediction result returned."
            )

        result = results[0]

        if len(result.boxes) == 0:

            output_path = save_default_image(
                image_path
            )

            gc.collect()

            return {
                "detected": False,
                "confidence": 0.0,
                "result_image": os.path.basename(
                    output_path
                ),
                "label": "No Ringworm Detected",
            }

        best_box = max(
            result.boxes,
            key=lambda box: float(box.conf[0])
        )

        confidence = float(
            best_box.conf[0]
        )

        output_path = save_annotated_image(
            image_path,
            result
        )

        gc.collect()

        return {
            "detected": True,
            "confidence": round(
                confidence,
                2
            ),
            "result_image": os.path.basename(
                output_path
            ),
            "label": "Ringworm",
        }

    except Exception as error:

        gc.collect()

        print(
            f"Inference error: {error}"
        )

        raise RuntimeError(
            f"Inference failed: {error}"
        )


def save_default_image(image_path):
    """Save original image when no ringworm is detected."""

    target_name = (
        f"result_{os.path.basename(image_path)}"
    )

    target_path = os.path.join(
        RESULTS_DIR,
        target_name
    )

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            "Unable to open uploaded image."
        )

    cv2.imwrite(
        target_path,
        image
    )

    return target_path


def save_annotated_image(
    image_path,
    result
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

        confidence = float(
            box.conf[0]
        )

        text = (
            f"Ringworm "
            f"{int(confidence * 100)}%"
        )

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
                y1 - 10
                if y1 - 10 > 20
                else y1 + 20
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

    cv2.imwrite(
        target_path,
        image
    )

    return target_path