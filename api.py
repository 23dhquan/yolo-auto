from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

# ========== NẠP MODEL YOLO ==========
MODEL_PATH = "best.pt"
print("🧠 Đang tải model YOLO...")
model = YOLO(MODEL_PATH)
print("✅ Model đã sẵn sàng!")

@app.route("/")
def home():
    return "YOLO API đang hoạt động 🚀"

# ========== API NHẬN ẢNH ==========
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file ảnh"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Tên file rỗng"}), 400

    # Lưu ảnh tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        img_path = tmp.name

    # Dự đoán YOLO
    results = model(img_path, conf=0.4, save=False)[0]
    boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x0, y0, x1, y1 = map(int, box.xyxy[0])
        boxes.append({
            "label": label,
            "confidence": round(conf, 2),
            "bbox": [x0, y0, x1, y1]
        })

    boxes.sort(key=lambda b: b["bbox"][0])
    boxes = boxes[:7]  # chỉ lấy 7 box đầu

    # Xóa file tạm
    os.remove(img_path)

    return jsonify({
        "count": len(boxes),
        "detections": boxes
    })

if __name__ == "__main__":
    # Render sẽ dùng cổng 10000 hoặc PORT env var
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
