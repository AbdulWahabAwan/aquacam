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
    # Check if YOLOv5 repository exists, clone if not
    if not os.path.exists('yolov5'):
        print("⚠️ YOLOv5 repository not found, cloning...")
        os.system('git clone https://github.com/ultralytics/yolov5.git')
        os.system('pip install -r yolov5/requirements.txt')
    
    model_path = "aquacam/best.pt"
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found at: {model_path}")
        # You might want to add a fallback here or exit gracefully
    
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local',force_reload=True)
    model.eval()
    print("✅ Model loaded successfully!")
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
        } for _, row in class_map.iterrows()  # Fixed syntax error here
    }
    print("✅ Class mapping loaded successfully!")
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
    
    annotated_img = results.render()[0]
    _, img_encoded = cv2.imencode('.jpg', annotated_img)  # Fixed variable name
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

if __name__ == "__main__":  # Fixed syntax error here
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
