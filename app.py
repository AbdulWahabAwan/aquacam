import os
import sys
import torch
import cv2
import base64
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import io

# Setup Flask
app = Flask(__name__)

# Add yolov5 repo path to sys.path
sys.path.append('yolov5')  # assumes yolov5 folder is in same directory

# Import YOLOv5 utils
from models.experimental import attempt_load

# 🔽 Function to download model from Google Drive
def download_model_from_drive():
    import gdown
    model_path = 'yolov5_backup/runs/train/exp/weights/best.pt'
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1iNWK4mrfZXpXU6-C4u89afgK0aqAp4CJ"
        gdown.download(url, model_path, quiet=False)

# 🔁 Download model if not already present
download_model_from_drive()

# 🔌 Load model
try:
    model = attempt_load('yolov5/best.pt', map_location=torch.device('cpu'))  # or 'cuda'
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# 📋 Load class mapping
try:
    class_map = pd.read_csv('class_mapping.csv')
    class_dict = {
        row['Folder Name']: {
            'species': row['Class Name'],
            'harmful': int(row['Harmful'])
        } for _, row in class_map.iterrows()
    }
    print("✅ Class mapping loaded")
except Exception as e:
    print(f"❌ Class mapping load failed: {e}")
    class_dict = {}

# 🔍 Health check
@app.route("/")
def home():
    return "✅ AquaCam Flask server is running!"

# 🧠 Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files['image']
        img_bytes = image_file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        results = model(image)
        predictions = results.pandas().xyxy[0].to_dict(orient="records")

        annotated_img = results.render()[0]
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

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# 🚀 App entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
