import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import pandas as pd
import base64
import cv2

app = Flask(__name__)
CORS(app)

# === Load YOLOv5 Model ===
try:
    model_path = "aquacam/best.pt"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='github')
    model.eval()
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# === Load class mapping ===
try:
    class_map = pd.read_csv('class_mapping.csv')
    class_dict = {
        row['Folder Name']: {
            'species': row['Class Name'],
            'harmful': int(row['Harmful'])
        } for _, row in class_map.iterrows()
    }
except Exception as e:
    print(f"❌ Class mapping load failed: {e}")
    class_dict = {}

@app.route("/")
def home():
    return "✅ AquaCam Flask server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    results = model(image)
    predictions = results.pandas().xyxy[0].to_dict(orient="records")

    annotated_img = results.render()[0]  # Already BGR
    _, img_encoded = cv2.imencode('.jpg', annotated_img)
    annotated_base64 = base64.b64encode(img_encoded).decode('utf-8')

    report = []
    for pred in predictions:
        label = pred['name']
        info = class_dict.get(label, {'species': 'Unknown', 'harmful': 'Unknown'})
        report.append({
            "code": label,
            "species": info['species'],
            "harmful": info['harmful'],
            "confidence": round(pred['confidence'] * 100, 2)
        })

    return jsonify({
        "annotated_image_base64": annotated_base64,
        "report": report
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
