from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import fitz

from .constants import UPLOAD_FOLDER, OUTPUT_FOLDER
from .table_detection import TableDetector, detect_tables
import cv2
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Initialize the table detection model
table_detector = TableDetector(max_side=2000)


def create_output_folder() -> str:
    """Create a unique timestamped output folder and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], timestamp)
    os.makedirs(output_path, exist_ok=True)
    return output_path


@app.route("/", methods=["GET"])
def index():
    """Return the upload HTML page."""
    return """
    <!doctype html>
    <html>
    <head>
      <title>Table Detector</title>
    </head>
    <body>
      <h1>Upload a single-page PDF</h1>
      <form method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
        <input type="file" name="file" accept="application/pdf" required>
        <input type="submit" value="Upload">
      </form>

      <div id="result">
        <img id="annotatedImage" style="max-width: 100%; margin-top: 20px;">
        <div id="caption" style="font-weight: bold; margin-top: 10px; color: red;"></div>
      </div>

      <script>
        const form = document.getElementById('uploadForm');
        form.onsubmit = async (e) => {
          e.preventDefault();
          const formData = new FormData(form);

          const res = await fetch('/upload', { method: 'POST', body: formData });
          const data = await res.json();

          if(data.error) {
            alert(data.error);
            return;
          }

          const img = document.getElementById('annotatedImage');
          const caption = document.getElementById('caption');

          img.src = "data:image/png;base64," + data.annotated_image_base64;

          if (data.num_tables === 0) {
            caption.textContent = "No tables detected";
          } else {
            caption.textContent = "";
          }
        }
      </script>
    </body>
    </html>
    """


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    Handle PDF upload, validate, detect tables, and return annotated image and results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)

    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Validate PDF has exactly 1 page
    doc = fitz.open(filepath)
    if len(doc) != 1:
        doc.close()
        return jsonify({"error": "PDF must contain exactly 1 page"}), 400
    doc.close()

    result = detect_tables(table_detector, filepath)
    output_folder = create_output_folder()

    json_path = os.path.join(output_folder, "results.json")
    with open(json_path, 'w') as f:
        json.dump({
            "num_tables": result["num_tables"],
            "tables": result["tables"]
        }, f, indent=2)

    if result.get("annotated_image") is not None:
        image_path = os.path.join(output_folder, "annotated_image.png")
        cv2.imwrite(image_path, result["annotated_image"])

    return jsonify({
        "annotated_image_base64": result["annotated_image_base64"],
        "num_tables": result["num_tables"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
