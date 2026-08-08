import os
import gc
import cv2
import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Limit PyTorch CPU threads to reduce Render memory usage
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best.pt"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = None
load_exception = None


def get_model():
    """
    Load the YOLO model only once.
    """

    global model
    global load_exception

    if model is not None:
        return model

    try:

        print("=" * 60)
        print("Loading Ringworm YOLO model")
        print(f"Model path: {MODEL_PATH}")
        print("=" * 60)

        # Make sure model exists
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}"
            )

        # Load model
        loaded_model = YOLO(MODEL_PATH)

        # CPU only
        loaded_model.to("cpu")

        # Evaluation mode
        loaded_model.model.eval()

        model = loaded_model

        print("YOLO model loaded successfully.")
        print("=" * 60)

        return model

    except Exception as error:

        load_exception = error

        gc.collect()

        raise RuntimeError(
            f"Unable to load model: {error}"
        )


# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------

def detect_ringworm(image_path):
    """
    Run ringworm detection using the trained YOLO model.
    Designed for low-memory Render instances.
    """

    detector = get_model()

    try:

        print(f"Processing image: {image_path}")

        # -------------------------------------------------
        # Run inference
        # -------------------------------------------------

        results = detector.predict(
            source=image_path,

            # Smaller image size to reduce RAM usage
            imgsz=320,

            # Detection confidence
            conf=0.25,

            # Limit detections
            max_det=3,

            # CPU
            device="cpu",

            # Disable half precision on CPU
            half=False,

            # Don't print huge YOLO output
            verbose=False,

            # Stream results
            stream=True
        )

        # Get first result
        result = next(iter(results))

        # -------------------------------------------------
        # No detection
        # -------------------------------------------------

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

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

        # -------------------------------------------------
        # Find highest confidence detection
        # -------------------------------------------------

        best_box = max(
            result.boxes,
            key=lambda box: float(
                box.conf[0]
            )
        )

        confidence = float(
            best_box.conf[0]
        )

        label = "Ringworm"

        # -------------------------------------------------
        # Save annotated image
        # -------------------------------------------------

        output_path = save_annotated_image(
            image_path,
            result,
            label
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
            "label": label,
        }

    except Exception as error:

        gc.collect()

        print(
            f"ERROR during inference: {error}"
        )

        raise RuntimeError(
            f"Inference failed: {error}"
        )


# ---------------------------------------------------------
# Save original image
# ---------------------------------------------------------

def save_default_image(image_path):
    """
    Save the original image when no ringworm
    is detected.
    """

    filename = os.path.basename(
        image_path
    )

    target_name = f"result_{filename}"

    target_path = os.path.join(
        RESULTS_DIR,
        target_name
    )

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            "Unable to open uploaded image."
        )

    success = cv2.imwrite(
        target_path,
        image
    )

    if not success:
        raise RuntimeError(
            "Unable to save result image."
        )

    return target_path


# ---------------------------------------------------------
# Save annotated image
# ---------------------------------------------------------

def save_annotated_image(
    image_path,
    result,
    label
):
    """
    Draw bounding boxes and confidence
    scores on the uploaded image.
    """

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise FileNotFoundError(
            "Unable to open uploaded image."
        )

    # Draw every detection
    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        confidence = float(
            box.conf[0]
        )

        text = (
            f"{label} "
            f"{int(confidence * 100)}%"
        )

        # Blue bounding box
        color = (255, 0, 0)

        thickness = 2

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # Put label above box
        text_y = (
            y1 - 10
            if y1 - 10 > 20
            else y1 + 25
        )

        cv2.putText(
            image,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    # Output filename
    filename = os.path.basename(
        image_path
    )

    target_name = f"result_{filename}"

    target_path = os.path.join(
        RESULTS_DIR,
        target_name
    )

    success = cv2.imwrite(
        target_path,
        image
    )

    if not success:
        raise RuntimeError(
            "Unable to save annotated result image."
        )

    return target_path