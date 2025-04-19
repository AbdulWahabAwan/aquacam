import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
import pandas as pd
import base64
import cv2
import sys

# Add local YOLOv5 repo to path (must be included in your repo)
sys.path.insert(0, './yolov5')  # Adjust if yolov5 is in a subfolder

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device

app = Flask(__name__)

# Load YOLOv5 model manually
try:
    device = select_device('')
    model_path = os.getenv('MODEL_PATH', 'best.pt')
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    print("✅ Model loaded successfully")
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
    print("✅ Class mapping loaded")
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
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess
    img_resized = letterbox(img, 640, stride=32, auto=True)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)[0]

    report = []
    for *xyxy, conf, cls in pred:
        label = model.names[int(cls)]
        info = class_dict.get(label, {'species': 'Unknown', 'harmful': 'Unknown'})
        report.append({
            "code": label,
            "species": info['species'],
            "harmful": info['harmful'],
            "confidence": round(conf.item() * 100, 2)
        })
        # Draw boxes
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    _, img_encoded = cv2.imencode('.jpg', img)
    annotated_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({
        "annotated_image_base64": annotated_base64,
        "report": report
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
