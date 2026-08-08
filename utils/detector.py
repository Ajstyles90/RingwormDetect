import os
import gc
import time

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "best.onnx")

RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Model metadata (verified from the exported ONNX graph):
#   Input  : images  -> [1, 3, 320, 320]  float32
#   Output : output0 -> [1, 5, 2100]      float32
#   2100 = 40^2 + 20^2 + 10^2 (3 YOLO scales)
#   5 = [cx, cy, w, h, class_conf]  ->  single class (ringworm)
#   Coordinates are already decoded to the 320x320 input pixel space.
INPUT_SIZE = 320
NUM_CLASSES = 1
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DET = 3
# Cap the longest side of the uploaded image before inference to limit RAM.
MAX_SOURCE_DIM = 1280


def _build_session_options():
    """CPU-efficient ONNX Runtime session options."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


# Load the ONNX session ONCE at worker startup (single model instance).
_session = None
_loaded = False


def get_session():
    """Return the lazily-created ONNX Runtime session (loaded once)."""
    global _session, _loaded

    if _loaded:
        return _session

    print(f"Loading ONNX model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    options = _build_session_options()

    _session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    print(f"ONNX Runtime {ort.__version__} | providers: {_session.get_providers()}")

    for inp in _session.get_inputs():
        print(f"Model input: name={inp.name} shape={inp.shape} type={inp.type}")
    for out in _session.get_outputs():
        print(f"Model output: name={out.name} shape={out.shape} type={out.type}")

    _loaded = True
    return _session


def _letterbox(image, size=320):
    """Resize with aspect ratio preserved and pad to a square canvas."""
    h, w = image.shape[:2]
    ratio = size / max(h, w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    return canvas, ratio, new_w, new_h


def _preprocess(image):
    """Convert an OpenCV BGR image to the model input blob."""
    canvas, ratio, new_w, new_h = _letterbox(image, INPUT_SIZE)

    # BGR -> RGB, uint8 -> float32, normalize to [0, 1]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0

    # HWC -> CHW, add batch dim -> [1, 3, 320, 320]
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    blob = np.ascontiguousarray(blob)

    return blob, ratio, new_w, new_h


def _nms(boxes, scores, iou_threshold=0.45):
    """Class-agnostic NMS. boxes: (N,4) xyxy, scores: (N,). Returns indices."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        order = order[1:][iou <= iou_threshold]

    return keep


def _postprocess(output, orig_w, orig_h, ratio, new_w, new_h):
    """Decode raw model output into original-image-space boxes."""
    # output shape: [1, 5, 2100] -> [2100, 5]
    preds = np.asarray(output[0]).reshape(5, -1).T.astype(np.float32)

    conf = preds[:, 4]

    mask = conf > CONF_THRESHOLD
    if not mask.any():
        return []

    preds = preds[mask]
    conf = conf[mask]

    cx = preds[:, 0]
    cy = preds[:, 1]
    bw = preds[:, 2]
    bh = preds[:, 3]

    # Convert cx,cy,w,h (in 320x320 letterbox space) to xyxy in original image.
    x1 = (cx - bw / 2.0) / ratio
    y1 = (cy - bh / 2.0) / ratio
    x2 = (cx + bw / 2.0) / ratio
    y2 = (cy + bh / 2.0) / ratio

    # Clamp to original image bounds.
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes, conf, IOU_THRESHOLD)

    # Sort kept detections by confidence, limit to MAX_DET.
    kept_scores = conf[keep]
    order = np.argsort(kept_scores)[::-1][:MAX_DET]
    keep = np.array(keep)[order]

    detections = []
    for idx in keep:
        x1v, y1v, x2v, y2v = boxes[idx]
        detections.append(
            {
                "x1": int(round(float(x1v))),
                "y1": int(round(float(y1v))),
                "x2": int(round(float(x2v))),
                "y2": int(round(float(y2v))),
                "confidence": float(conf[idx]),
            }
        )

    return detections


def _load_image(image_path):
    """Load an image, capping its largest dimension to limit RAM."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Unable to open uploaded image.")

    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim > MAX_SOURCE_DIM:
        scale = MAX_SOURCE_DIM / max_dim
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def detect_ringworm(image_path):
    """Run ONNX object detection and return the expected result dict."""
    inference_time = None

    try:
        image = _load_image(image_path)
        orig_h, orig_w = image.shape[:2]

        print(f"Processing image: {image_path} (size {orig_w}x{orig_h})")

        session = get_session()
        input_name = session.get_inputs()[0].name

        blob, ratio, new_w, new_h = _preprocess(image)

        print("Starting inference...")
        start = time.time()
        output = session.run(None, {input_name: blob})[0]
        inference_time = time.time() - start
        print(f"Inference completed in {inference_time:.3f} seconds")

        detections = _postprocess(output, orig_w, orig_h, ratio, new_w, new_h)
        print(f"Number of detections: {len(detections)}")

        if not detections:
            output_path = save_default_image(image_path)
            return {
                "detected": False,
                "confidence": 0.0,
                "result_image": os.path.basename(output_path),
                "label": "No Ringworm Detected",
            }

        best = detections[0]

        output_path = save_annotated_image(image_path, detections)

        return {
            "detected": True,
            "confidence": round(best["confidence"], 2),
            "result_image": os.path.basename(output_path),
            "label": "Ringworm",
        }

    except Exception as error:
        print(f"Inference error: {error}")
        raise RuntimeError(f"Inference failed: {error}")

    finally:
        del image
        if inference_time is None:
            gc.collect()


def save_default_image(image_path):
    """Save the original image when nothing is detected."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Unable to open uploaded image.")

    target_name = f"result_{os.path.basename(image_path)}"
    target_path = os.path.join(RESULTS_DIR, target_name)

    cv2.imwrite(target_path, image)

    return target_path


def save_annotated_image(image_path, detections):
    """Draw detection boxes and labels, then save the result."""
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Unable to open uploaded image.")

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        confidence = det["confidence"]

        text = f"Ringworm {int(confidence * 100)}%"
        color = (255, 0, 0)
        thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        cv2.putText(
            img,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    target_name = f"result_{os.path.basename(image_path)}"
    target_path = os.path.join(RESULTS_DIR, target_name)

    cv2.imwrite(target_path, img)

    return target_path
