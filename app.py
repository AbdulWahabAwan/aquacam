from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import pandas as pd
import base64
import cv2

app = Flask(__name__)

# Load model
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.eval()
except Exception as e:
    print(f"❌ Model load failed: {e}")


# Load class mapping
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
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    img_bytes = image_file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

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
    app.run(host="0.0.0.0", port=10000)
