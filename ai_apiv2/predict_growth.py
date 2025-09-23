from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
import schedule
import time
import datetime
from flask import request, jsonify
import base64
from io import BytesIO


def image_to_base64(image, format="PNG"):
    """
    Convert PIL.Image or NumPy image to base64 string.
    """
    if isinstance(image, np.ndarray):
        # If NumPy → Convert to PIL first
        image = Image.fromarray(image.astype(np.uint8))

    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string


def predictiongrowth(image, weights):
    # Load a pretrained YOLOv8n model
    model = YOLO(weights)
    results = model.predict(source=image, conf=0.8)
    result = results[0]

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    detections = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = result.names[class_id]
        text = f"{label} {conf:.2f}"

        # วาดกรอบ
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

        # วาดพื้นหลังข้อความ
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="blue")

        draw.text((x1 + 2, y1 - th - 2), text, fill="white", font=font)

        detections.append(
            {
                "label": label,
                "conf": round(conf, 4),
                # "bbox": [x1, y1, x2, y2]
            }
        )

    # 🔁 แปลงภาพผลลัพธ์กลับเป็น base64
    b64 = image_to_base64(image)

    # 📦 รวมทั้งหมดเป็น JSON
    result_json = {"detections": detections, "image_base64": b64}
    return result_json


def predict_growth(image):

    try:
        weights = "./growth.pt"
        result_json = predictiongrowth(image, weights)
        # b64 = image_to_base64(masked_img_pil)
        # result_json.append({"base64":b64})
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
