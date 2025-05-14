from flask import Flask, request, jsonify
import torch
import io
from PIL import Image
import base64
import os

app = Flask(__name__)

# Force hub cache to be in local dir (avoids using system path with OpenCV issue)
torch.hub.set_dir('./torch_cache')

# Load YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='best.pt',
    force_reload=True  # ensure a clean download
)
model.eval()

@app.route('/')
def index():
    return "YOLOv5 Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    img_bytes = image_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    results = model(img)
    results_dict = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({'results': results_dict})

if __name__ == '__main__':
    app.run(debug=True)
