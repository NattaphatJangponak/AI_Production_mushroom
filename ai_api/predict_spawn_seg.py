# predict_spawn_seg.py

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import base64
from io import BytesIO
from flask import jsonify
import math
import random

# โหลดโมเดล segmentation (เน้นถุง)
model = YOLO("pot_seg.pt")  # ← เปลี่ยนตาม path ของคุณถ้าใช้โมเดลอื่น


def image_to_base64(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def predict_spawn_seg(image_pil):
    try:
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        results = model.predict(image_cv2, conf=0.4, save=False)

        for r in results:
            img = r.orig_img.copy()
            h_img, w_img, _ = img.shape

            if r.masks is None:
                return jsonify({"error": "ไม่พบ segmentation mask"}), 200

            masks = r.masks.data.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            font_scale = max(1, min(h_img, w_img) / 600)
            font_thickness = max(1, int(font_scale * 2))

            instances = {"bag": [], "white": []}

            for mask, cls_id in zip(masks, cls_ids):
                # สมมุติว่า: class 3 = bag, class 6 = white/spawn (ปรับตาม model.names ของคุณ)
                if cls_id not in [3, 6]:
                    continue
                mask_resized = cv2.resize(
                    mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized.astype(bool)
                area = np.sum(mask_bool)
                y, x = np.where(mask_bool)
                if len(x) == 0 or len(y) == 0:
                    continue
                cx, cy = int(np.mean(x)), int(np.mean(y))

                entry = {"mask": mask_bool, "area": area, "center": (cx, cy)}
                if cls_id == 3:
                    instances["bag"].append(entry)
                elif cls_id == 6:
                    instances["white"].append(entry)

            color_mask = np.zeros_like(img, dtype=np.uint8)

            for white in instances["white"]:
                cx_n, cy_n = white["center"]
                closest_bag = min(
                    instances["bag"],
                    key=lambda b: math.hypot(
                        cx_n - b["center"][0], cy_n - b["center"][1]
                    ),
                    default=None,
                )

                if closest_bag and closest_bag["area"] > 0:
                    white_area = white["area"]
                    bag_area = closest_bag["area"]
                    percent = (white_area / (white_area + bag_area)) * 100
                    percent = min(percent, 100.0)

                    text = f"white: {percent:.1f}%"
                    cx, cy = white["center"]
                    cv2.putText(
                        img,
                        text,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                    )
                    # cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

                color = [random.randint(100, 255) for _ in range(3)]
                color_mask[white["mask"]] = color

            for bag in instances["bag"]:
                color = [random.randint(100, 255) for _ in range(3)]
                color_mask[bag["mask"]] = color
                cx, cy = bag["center"]
                cv2.putText(
                    img,
                    "bag",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    font_thickness,
                )

            blended = cv2.addWeighted(img, 0.7, color_mask, 0.5, 0)
            blended_pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            b64 = image_to_base64(blended_pil)
            # คำนวณเปอร์เซ็นต์
            white_area = white["area"]
            bag_area = closest_bag["area"]
            percent = (white_area / (white_area + bag_area)) * 100
            text = f"{percent:.2f}"

            return [b64, text]

        return ""
    except Exception as e:
        return ""
