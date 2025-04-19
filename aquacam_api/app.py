from flask import Flask, request, jsonify
import torch
import pandas as pd
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)
model.eval()

mapping_df = pd.read_csv('class_mapping.csv')
class_map = {row['code']: {"species": row['species'], "harmful": bool(row['harmful'])} for _, row in mapping_df.iterrows()}

@app.route("/", methods=["GET"])
def index():
    return "✅ AquaCam API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)
    preds = results.pandas().xyxy[0].to_dict(orient="records")

    output = []
    for pred in preds:
        class_name = pred['name']
        confidence = float(pred['confidence'])

        mapped = class_map.get(class_name, {"species": class_name, "harmful": False})

        output.append({
            "species": mapped['species'],
            "harmful": mapped['harmful'],
            "confidence": confidence,
            "bbox": [
                float(pred['xmin']), float(pred['ymin']),
                float(pred['xmax']), float(pred['ymax'])
            ]
        })

    results.render()
    annotated_image = Image.fromarray(results.ims[0])
    buffered = io.BytesIO()
    annotated_image.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "predictions": output,
        "annotated_image_base64": encoded_img
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))