from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import random


# โหลดโมเดล YOLOv8 (Segmentation) — ปรับ path ให้ตรงกับน้ำหนักโมเดลของคุณ
# ควรเป็นโมเดลที่เทรนสำหรับงาน growth segmentation ที่มีคลาส: 0=abnormal, 1=normal
MODEL_PATH = "grow_seg.pt"
model = YOLO(MODEL_PATH)


def image_to_base64(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def predict_growth_seg(image_pil):
    try:
        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # ปรับการตั้งค่าเพื่อเพิ่มความไวในการตรวจจับ
        # results = model.predict(
        #     source=image_cv2,
        #     conf=0.15,
        #     iou=0.3,
        #     max_det=100,
        #     agnostic_nms=False,
        #     save=False,
        # )
        results = model.predict(
            source=image_cv2,
            conf=0.15,
            iou=0.3,
            max_det=100,
            agnostic_nms=False,
            save=False,
        )

        for r in results:
            img = r.orig_img.copy()
            h_img, w_img, _ = img.shape

            if r.masks is None:
                # ไม่พบ mask — ส่งภาพเดิมกลับ
                bgr_fallback = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                pil_fallback = Image.fromarray(
                    cv2.cvtColor(bgr_fallback, cv2.COLOR_BGR2RGB)
                )
                return image_to_base64(pil_fallback)

            masks = r.masks.data.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []

            print("\n📊 ผลลัพธ์การ detection:")
            print(f"  - จำนวน masks ที่พบ: {len(masks)}")
            if len(confidences) > 0:
                print(
                    "  - confidence scores:",
                    [f"{c:.3f}" for c in confidences],
                )

            instances = {"abnormal": [], "normal": []}

            for i, (mask, cls_id) in enumerate(zip(masks, cls_ids)):
                if cls_id not in [0, 1]:
                    continue

                mask_resized = cv2.resize(
                    mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized.astype(bool)
                area = int(np.sum(mask_bool))

                # กรองขนาดเล็กมาก
                if area < 100:
                    print(f"  ⚠️ ข้าม detection {i+1}: พื้นที่เล็กเกินไป ({area} pixels)")
                    continue

                y, x = np.where(mask_bool)
                if len(x) == 0 or len(y) == 0:
                    continue
                cx, cy = int(np.mean(x)), int(np.mean(y))

                entry = {
                    "mask": mask_bool,
                    "area": area,
                    "center": (cx, cy),
                    "id": i + 1,
                }

                if cls_id == 0:
                    instances["abnormal"].append(entry)
                    print(
                        f"  ✅ Abnormal #{len(instances['abnormal'])}: area={area}, center=({cx},{cy})"
                    )
                else:
                    instances["normal"].append(entry)
                    print(
                        f"  ✅ Normal #{len(instances['normal'])}: area={area}, center=({cx},{cy})"
                    )

            print(
                f"\n[DEBUG] ผลลัพธ์สุดท้าย: abnormal {len(instances['abnormal'])} จุด, normal {len(instances['normal'])} ดอก"
            )

            # สร้าง mask สี
            color_mask = np.zeros_like(img, dtype=np.uint8)

            # รวมพื้นที่
            total_normal_area = (
                sum(n["area"] for n in instances["normal"])
                if instances["normal"]
                else 0
            )
            total_abnormal_area = (
                sum(a["area"] for a in instances["abnormal"])
                if instances["abnormal"]
                else 0
            )
            total_area = total_normal_area + total_abnormal_area

            has_abnormal = len(instances["abnormal"]) > 0

            # ถ้าไม่มี abnormal ให้โชว์ normal
            if not has_abnormal:
                for i, normal in enumerate(instances["normal"], 1):
                    color_mask[normal["mask"]] = [0, 255, 0]
                    cx, cy = normal["center"]
                    cv2.putText(
                        img,
                        f"N{i}",
                        (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # โชว์ abnormal เสมอ
            for i, abnormal in enumerate(instances["abnormal"], 1):
                color_mask[abnormal["mask"]] = [0, 0, 255]
                cx, cy = abnormal["center"]
                cv2.putText(
                    img,
                    f"A{i}",
                    (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), 3)

            blended = cv2.addWeighted(img, 0.7, color_mask, 0.5, 0)
            blended_pil = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

            safe = True

            # สรุปผลแบบ log
            if has_abnormal:
                safe = False
                abnormal_percent = (
                    (total_abnormal_area / total_area) * 100 if total_area > 0 else 0
                )
                print("\n" + "=" * 60)
                print("🍄 สรุปผลการวิเคราะห์ดอกเห็ด (Enhanced Detection)")
                print("=" * 60)
                print("🔴 สถานะ: ผิดปกติ (ABNORMAL)")
                print(f"📊 เปอร์เซ็นต์ที่ผิดปกติ: {abnormal_percent:.1f}%")
                print(f"🔢 จำนวนส่วนที่ผิดปกติ: {len(instances['abnormal'])} จุด")
                print(f"🔢 จำนวนส่วนที่ปกติ: {len(instances['normal'])} ดอก")
                print("\n📋 รายละเอียด Abnormal segments:")
                for i, abnormal in enumerate(instances["abnormal"], 1):
                    area = abnormal["area"]
                    print(f"  A{i}: area={area:,} pixels")
            else:
                normal_percent = (
                    (total_normal_area / total_area) * 100 if total_area > 0 else 0
                )
                print("\n" + "=" * 60)
                print("🍄 สรุปผลการวิเคราะห์ดอกเห็ด (Enhanced Detection)")
                print("=" * 60)
                print("🟢 สถานะ: ปกติ (NORMAL)")
                print(f"📊 เปอร์เซ็นต์ที่ปกติ: {normal_percent:.1f}%")
                print(f"🔢 จำนวนส่วนที่ปกติ: {len(instances['normal'])} ดอก")

            print(f"📏 พื้นที่รวมทั้งหมด: {total_area:,} pixels")

            if len(instances["abnormal"]) < 3:
                print("\n💡 หมายเหตุ: หากต้องการหาจุด abnormal เพิ่มเติม ลองปรับค่า:")
                print("  - ลด conf เป็น 0.1 หรือ 0.05")
                print("  - ลด iou เป็น 0.2")
                print("  - ตรวจสอบการ label ใน dataset")
            print("=" * 60)

            # ส่งกลับเฉพาะภาพผลลัพธ์เป็น base64 (สอดคล้องกับสไตล์ของ predict_spawn_seg)
            if safe:
                return [image_to_base64(blended_pil), "Normal"]
            else:
                return [image_to_base64(blended_pil), "Abnormal"]

        return ""
    except Exception:
        return ""
