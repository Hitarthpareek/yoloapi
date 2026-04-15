from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

# 🔥 load model once (important)
model = YOLO("model/best.pt")

@app.route("/")
def home():
    return "YOLO API Running 🚀"

@app.route("/yolo", methods=["POST"])
def detect():
    try:
        data = request.json
        image = data.get("image")

        if not image:
            return jsonify({"error": "No image provided"}), 400

        # decode base64
        img_data = base64.b64decode(image.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 🚀 run YOLO
        results = model(img)[0]

        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "type": model.names[cls],
                "confidence": conf,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1
            })

        return jsonify({ "boxes": detections })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500


# 🔥 important for deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
