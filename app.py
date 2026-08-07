import os
import time
import threading

try:
    import webview
except ImportError:
    webview = None
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from utils.detector import detect_ringworm


# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


# Create Flask app
app = Flask(__name__)

app.secret_key = "ringworm-secret-key"

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
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, run detection, and display results."""

    if "image" not in request.files:
        flash("No image file part in the request.")
        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "":
        flash("Please choose an image before submitting.")
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        flash(
            "Unsupported file type. Please upload PNG, JPG, JPEG or GIF."
        )
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(upload_path)

    # Run detection
    start_time = time.time()

    try:
        result = detect_ringworm(upload_path)

    except Exception as error:
        flash(f"Model error: {error}")
        return redirect(url_for("home"))

    end_time = time.time()

    result["time_seconds"] = round(
        end_time - start_time,
        2
    )

    return render_template(
        "index.html",
        original_image=url_for(
            "static",
            filename=f"uploads/{filename}"
        ),
        result_image=url_for(
            "static",
            filename=f"results/{result['result_image']}"
        ),
        detected=result["detected"],
        label=result["label"],
        confidence=result["confidence"],
        time_seconds=result["time_seconds"],
    )


def start_flask():
    """Start Flask for the desktop application."""
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":

    if webview is not None:

        flask_thread = threading.Thread(
            target=start_flask,
            daemon=True
        )

        flask_thread.start()

        webview.create_window(
            "Ringworm Detection System",
            "http://127.0.0.1:5000",
            width=1200,
            height=800,
            resizable=True
        )

        webview.start()

    else:

        app.run(
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)),
            debug=False
        )