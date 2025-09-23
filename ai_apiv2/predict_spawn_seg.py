# predict_spawn_seg.py

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import base64
from io import BytesIO
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


def calculate_bounding_box(mask_bool):
    """
    คำนวณกรอบรอบ mask

    Args:
        mask_bool: mask ของวัตถุ

    Returns:
        tuple: (x1, y1, x2, y2) หรือ None ถ้าไม่พบ
    """
    y, x = np.where(mask_bool)
    if len(x) == 0 or len(y) == 0:
        return None

    x1 = int(np.min(x))
    y1 = int(np.min(y))
    x2 = int(np.max(x))
    y2 = int(np.max(y))

    return x1, y1, x2, y2


def draw_frame(img, x1, y1, x2, y2, color=(0, 0, 255), thickness=5):
    """
    วาดกรอบรอบพื้นที่ที่เลือก พร้อมกากบาทกลาง
    """
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.line(img, (x1, center_y), (x2, center_y), color, thickness)
    cv2.line(img, (center_x, y1), (center_x, y2), color, thickness)


def draw_9_grid(img, color=(128, 128, 128), thickness=2):
    """
    วาดตาราง 9 ช่อง (3x3) บนภาพ
    """
    h, w = img.shape[:2]
    x1 = w // 3
    x2 = 2 * w // 3
    y1 = h // 3
    y2 = 2 * h // 3
    cv2.line(img, (x1, 0), (x1, h), color, thickness)
    cv2.line(img, (x2, 0), (x2, h), color, thickness)
    cv2.line(img, (0, y1), (w, y1), color, thickness)
    cv2.line(img, (0, y2), (w, y2), color, thickness)


def check_grid_intersection(x1, y1, x2, y2, img_shape):
    """
    ตรวจสอบว่ากรอบ (x1, y1, x2, y2) ชนกับเส้นตาราง 9 ช่องกี่เส้น
    """
    h, w = img_shape[:2]
    grid_x1 = w // 3
    grid_x2 = 2 * w // 3
    grid_y1 = h // 3
    grid_y2 = 2 * h // 3

    intersections = 0
    if x1 <= grid_x1 <= x2:
        intersections += 1
    if x1 <= grid_x2 <= x2:
        intersections += 1
    if y1 <= grid_y1 <= y2:
        intersections += 1
    if y1 <= grid_y2 <= y2:
        intersections += 1

    return intersections


def predict_spawn_seg(image_pil):
    try:
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        results = model.predict(image_cv2, conf=0.4, save=False)

        final_percent = None
        final_b64 = None

        for r in results:
            img = r.orig_img.copy()
            h_img, w_img, _ = img.shape

            if r.masks is None:
                continue

            # วาดตาราง 9 ช่องก่อน
            draw_9_grid(img, color=(128, 128, 128), thickness=2)

            masks = r.masks.data.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            font_scale = max(1, min(h_img, w_img) / 600)
            font_thickness = max(1, int(font_scale * 2))

            instances = {"bag": [], "normal": []}

            for mask, cls_id in zip(masks, cls_ids):
                if cls_id not in [3, 6]:  # 3 = bag, 6 = normal (spawn/white)
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

                bbox = calculate_bounding_box(mask_bool)
                intersects_y1 = False
                intersects_y2 = False
                if bbox:
                    bx1, by1, bx2, by2 = bbox
                    grid_y1 = h_img // 3
                    grid_y2 = 2 * h_img // 3
                    intersects_y1 = by1 <= grid_y1 <= by2
                    intersects_y2 = by1 <= grid_y2 <= by2

                entry = {
                    "mask": mask_bool,
                    "area": area,
                    "center": (cx, cy),
                    "bbox": bbox,
                    "intersects_y1": intersects_y1,
                    "intersects_y2": intersects_y2,
                }
                if cls_id == 3:
                    # เฉพาะ bag ที่ชนกับ Y1 หรือ Y2
                    if intersects_y1 or intersects_y2:
                        instances["bag"].append(entry)
                elif cls_id == 6:
                    instances["normal"].append(entry)

            # สีสำหรับผลลัพธ์
            color_mask = np.zeros_like(img, dtype=np.uint8)

            # เรียง normal โดยให้ตัวที่ชน Y2 มาก่อน
            normal_sorted = sorted(
                instances["normal"], key=lambda x: x["intersects_y2"], reverse=True
            )

            # แยก bag ตามเส้น
            bag_y2 = [b for b in instances["bag"] if b["intersects_y2"]]
            bag_y1 = [
                b
                for b in instances["bag"]
                if b["intersects_y1"] and not b["intersects_y2"]
            ]

            if len(bag_y2) > 0:
                bag_sorted = sorted(
                    bag_y2, key=lambda x: x["intersects_y2"], reverse=True
                )
            elif len(bag_y1) > 0:
                bag_sorted = sorted(
                    bag_y1, key=lambda x: x["intersects_y1"], reverse=True
                )
            else:
                bag_sorted = []

            segmentation_done = False

            computed_percents = []

            for normal in normal_sorted:
                if segmentation_done:
                    break

                cx_n, cy_n = normal["center"]
                min_dist = float("inf")
                closest_bag = None
                for bag in instances["bag"]:
                    cx_b, cy_b = bag["center"]
                    dist = math.hypot(cx_n - cx_b, cy_n - cy_b)
                    if dist < min_dist:
                        min_dist = dist
                        closest_bag = bag

                percent = 0.0
                if closest_bag and closest_bag["area"] > 0:
                    normal_area = normal["area"]
                    bag_area = closest_bag["area"]
                    percent = (normal_area / (normal_area + bag_area)) * 100
                    percent = min(percent, 100.0)

                    cx, cy = normal["center"]
                    text = f"normal: {percent:.1f}%"
                    cv2.putText(
                        img,
                        text,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                    )

                # เงื่อนไข segmentation
                should_segment = False

                if (
                    percent > 90
                    and closest_bag
                    and not closest_bag.get("drawn_combined", False)
                ):
                    combined_mask = normal["mask"] | closest_bag["mask"]
                    bbox_combined = calculate_bounding_box(combined_mask)
                    if bbox_combined:
                        x1, y1, x2, y2 = bbox_combined
                        intersections = check_grid_intersection(
                            x1, y1, x2, y2, img.shape
                        )
                        if intersections >= 3:
                            should_segment = True
                            segmentation_done = True
                            draw_frame(
                                img, x1, y1, x2, y2, color=(0, 255, 255), thickness=4
                            )
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            text_combined = (
                                f"Combined: {percent:.1f}% ({intersections} lines)"
                            )
                            cv2.putText(
                                img,
                                text_combined,
                                (center_x, center_y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                (0, 0, 0),
                                font_thickness,
                            )
                            closest_bag["drawn_combined"] = True

                if (
                    not should_segment
                    and closest_bag
                    and not closest_bag.get("drawn_combined", False)
                ):
                    bbox = calculate_bounding_box(closest_bag["mask"])
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        intersections = check_grid_intersection(
                            x1, y1, x2, y2, img.shape
                        )
                        if intersections >= 3:
                            should_segment = True
                            segmentation_done = True

                if should_segment:
                    color = [random.randint(100, 255) for _ in range(3)]
                    color_mask[normal["mask"]] = color

                computed_percents.append(percent)

            # ใส่ชื่อและสีให้ bag ที่ผ่านเงื่อนไข
            if len(bag_sorted) > 0:
                for bag in bag_sorted:
                    should_segment_bag = False
                    bbox = bag["bbox"]
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        intersections = check_grid_intersection(
                            x1, y1, x2, y2, img.shape
                        )
                        if intersections >= 3:
                            should_segment_bag = True
                    if should_segment_bag:
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

            # เลือก percent ที่ดีที่สุดเพื่อส่งกลับ (ถ้าไม่มีคำนวณได้ ให้เป็น 0.0)
            best_percent = max(computed_percents) if computed_percents else 0.0
            final_percent = f"{best_percent:.2f}"
            final_b64 = b64

        if final_b64 is not None and final_percent is not None:
            return [final_b64, final_percent]
        else:
            return ""
    except Exception:
        return ""
