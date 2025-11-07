from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2, numpy as np, tempfile, os

app = Flask(__name__)
CORS(app)  # cho phép tất cả domain (hoặc giới hạn origin nếu muốn)

# ========== NẠP MODEL YOLO ==========
MODEL_PATH = "best.pt"
print("🧠 Đang tải model YOLO...")
model = YOLO(MODEL_PATH)
print("✅ Model đã sẵn sàng!")

@app.route("/")
def home():
    return "YOLO API đang hoạt động 🚀"

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
    boxes = boxes[:7]

    os.remove(img_path)

    return jsonify({
        "count": len(boxes),
        "detections": boxes
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
