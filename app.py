import os
import time
import threading

try:
    import webview
except ImportError:
    webview = None

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from utils.detector import detect_ringworm


# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


# Create Flask app
app = Flask(__name__)

# Use an environment-provided secret key in production; fall back for local dev.
app.secret_key = os.environ.get("SECRET_KEY", "ringworm-dev-secret-key")

app.config["UPLOAD_FOLDER"] = os.path.join(
    app.root_path, "static", "uploads"
)

app.config["RESULT_FOLDER"] = os.path.join(
    app.root_path, "static", "results"
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# Make sure required folders exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)


def allowed_file(filename):
    """Check whether the uploaded file has a valid extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def home():
    """Render the home page."""
    return render_template("home.html")


@app.route("/detection", methods=["GET"])
def detection():
    """Render the detection page."""
    return render_template("detection.html")


@app.route("/about", methods=["GET"])
def about():
    """Render the about page."""
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Handle image upload, run detection, and return JSON results."""
    if "image" not in request.files:
        return jsonify(
            {"success": False, "error": "No image was provided. Please choose an image to analyze."}
        ), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify(
            {"success": False, "error": "Please choose an image before submitting."}
        ), 400

    if not allowed_file(file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Unsupported file type. Please upload a PNG, JPG, JPEG, WEBP or GIF image.",
            }
        ), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    start_time = time.time()

    try:
        result = detect_ringworm(upload_path)
    except FileNotFoundError as error:
        return jsonify(
            {"success": False, "error": "The detection model is currently unavailable. Please try again later."}
        ), 503
    except Exception as error:
        app.logger.error("Detection failed: %s", error)
        return jsonify(
            {"success": False, "error": "Unable to analyze image. Something went wrong while processing the image. Please try again."}
        ), 500

    end_time = time.time()

    result["time_seconds"] = round(end_time - start_time, 2)
    result["success"] = True
    result["result_image_url"] = url_for(
        "static", filename=f"results/{result['result_image']}"
    )

    return jsonify(result)


def start_flask():
    """Start Flask for the desktop application."""
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":

    if webview is not None:

        flask_thread = threading.Thread(
            target=start_flask,
            daemon=True,
        )

        flask_thread.start()

        webview.create_window(
            "Ringworm Detection System",
            "http://127.0.0.1:5000",
            width=1200,
            height=800,
            resizable=True,
        )

        webview.start()

    else:

        # Bind to 0.0.0.0 and use PORT for Render / cloud hosting.
        app.run(
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)),
            debug=False,
        )
