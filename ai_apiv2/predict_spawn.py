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


def prediction(image, weights):
    # Load a pretrained YOLOv8n model
    model = YOLO(weights)

    # Replace 'your_image.jpg' with the actual path to your image file
    results = model.predict(
        source=image, conf=0.8
    )  # save=True saves the image with detections

    for result in results:
        if result.boxes is not None:
            for box in result.boxes[0]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxx = box.xyxy[0].tolist()
                # print(boxx[2])
                print(
                    f"Bounding Box: (x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f})"
                )
                confidence = box.conf[0].item()
                class_id = box.cls[0].item()
                label = model.names[int(box.cls[0].item())]

                # print(f"  Confidence: {confidence:.2f}")
                print(f"  Class ID: {class_id}")
                print(label)
                confidence = f"{confidence:.2f}"
                print(confidence)

    return boxx, confidence


def preprocess_image(img_cv2):
    """
    ทำการ Pre-processing ภาพเพื่อลดแสงสะท้อนสีขาว

    Args:
        img_cv2 (numpy.ndarray): ภาพในรูปแบบ OpenCV (BGR)

    Returns:
        numpy.ndarray: ภาพที่ผ่านการ Pre-processing แล้ว (BGR)
    """
    # Bilateral Filter (ลองปรับค่า d, sigmaColor, sigmaSpace)
    blurred_img = cv2.bilateralFilter(img_cv2, d=9, sigmaColor=75, sigmaSpace=75)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # กำหนดเกณฑ์สำหรับบริเวณที่อาจเป็นแสงสะท้อน (ค่า Value สูง)
    threshold_value = 200
    highlight_mask = v > threshold_value

    # ลดค่า Value ในบริเวณที่อาจเป็นแสงสะท้อน
    v[highlight_mask] = np.clip(v[highlight_mask] - 30, 0, 255)  # ปรับลดค่าตามความเหมาะสม

    merged_hsv = cv2.merge([h, s, v])
    preprocessed_img = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
    return preprocessed_img


def gray_world_white_balance(img_cv2):
    """
    ทำการปรับ White Balance โดยใช้วิธี Gray World Assumption

    Args:
        img_cv2 (numpy.ndarray): ภาพในรูปแบบ OpenCV (BGR)

    Returns:
        numpy.ndarray: ภาพที่ผ่านการปรับ White Balance แล้ว (BGR)
    """
    if len(img_cv2.shape) == 3:  # ตรวจสอบว่ามี 3 Channels (BGR/RGB)
        img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Grayscale Image", img_gray)
    elif len(img_cv2.shape) == 2:  # เป็นภาพ Grayscale อยู่แล้ว
        print("ภาพเป็น Grayscale อยู่แล้ว")
        # cv2.imshow("Grayscale Image", img_cv2)
    else:
        print("รูปแบบภาพไม่รองรับ")
    avg_gray = np.mean(img_gray)

    img_b, img_g, img_r = cv2.split(img_cv2.astype(np.float32))

    avg_b = np.mean(img_b)
    avg_g = np.mean(img_g)
    avg_r = np.mean(img_r)

    scale_b = avg_gray / avg_b if avg_b > 0 else 1
    scale_g = avg_gray / avg_g if avg_g > 0 else 1
    scale_r = avg_gray / avg_r if avg_r > 0 else 1

    balanced_b = np.clip(img_b * scale_b, 0, 255).astype(np.uint8)
    balanced_g = np.clip(img_g * scale_g, 0, 255).astype(np.uint8)
    balanced_r = np.clip(img_r * scale_r, 0, 255).astype(np.uint8)

    balanced_img = cv2.merge([balanced_b, balanced_g, balanced_r])
    return balanced_img


def selective_color_correction(img_cv2):
    """
    พยายามปรับสีขาวของแสงสะท้อนให้กลายเป็นสีด้านข้างของวัตถุ

    Args:
        img_cv2 (numpy.ndarray): ภาพในรูปแบบ OpenCV (BGR)

    Returns:
        numpy.ndarray: ภาพที่ผ่านการปรับสีแล้ว (BGR)
    """
    img_cv2 = np.array(img_cv2)
    hsv_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # 1. สร้าง Mask สำหรับบริเวณที่อาจเป็นแสงสะท้อนสีขาว (ปรับเกณฑ์ตามความเหมาะสม)
    lower_white_hsv = np.array([0, 0, 200])
    upper_white_hsv = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv_img, lower_white_hsv, upper_white_hsv)
    white_mask = cv2.dilate(
        white_mask, np.ones((5, 5), np.uint8), iterations=2
    )  # ขยาย Mask เล็กน้อย

    corrected_img = img_cv2.copy()
    for y in range(img_cv2.shape[0]):
        for x in range(img_cv2.shape[1]):
            if white_mask[y, x] > 0:
                # 2. หา "สีข้างเคียง" (วิธีง่ายๆ: สุ่มจากบริเวณใกล้เคียง)
                neighbor_colors = []
                for i in range(max(0, y - 5), min(img_cv2.shape[0], y + 6)):
                    for j in range(max(0, x - 5), min(img_cv2.shape[1], x + 6)):
                        if (
                            not (
                                max(0, y - 2) <= i <= min(img_cv2.shape[0] - 1, y + 2)
                                and max(0, x - 2)
                                <= j
                                <= min(img_cv2.shape[1] - 1, x + 2)
                            )
                            and white_mask[i, j] == 0
                        ):
                            neighbor_colors.append(img_cv2[i, j])

                if neighbor_colors:
                    # แทนที่สีขาวด้วยสีข้างเคียงแบบสุ่ม
                    corrected_img[y, x] = neighbor_colors[
                        np.random.randint(len(neighbor_colors))
                    ]
                else:
                    # หากไม่มีสีข้างเคียง ให้ใช้สีเดิม (อาจปรับให้มืดลงเล็กน้อย)
                    b, g, r = img_cv2[y, x].astype(float)
                    corrected_img[y, x] = np.clip(
                        [b * 0.8, g * 0.8, r * 0.8], 0, 255
                    ).astype(np.uint8)

    return corrected_img


def add_gaussian_noise(img_cv2, mean=0, stddev=20):
    """
    เพิ่ม Gaussian noise ลงในภาพ OpenCV

    Args:
        img_cv2 (numpy.ndarray): ภาพในรูปแบบ OpenCV (BGR)
        mean (float): ค่าเฉลี่ยของ Gaussian distribution
        stddev (float): ส่วนเบี่ยงเบนมาตรฐานของ Gaussian distribution

    Returns:
        numpy.ndarray: ภาพที่มี Gaussian noise (BGR)
    """
    noise = np.zeros(img_cv2.shape, np.int16)
    cv2.randn(noise, mean, stddev)
    noisy_img = cv2.add(img_cv2.astype(np.int16), noise)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def remove_gaussian_noise_gaussian_blur(img_cv2, kernel_size=(5, 5)):
    """
    ลด Gaussian noise โดยใช้ Gaussian Blur

    Args:
        img_cv2 (numpy.ndarray): ภาพที่มี Gaussian noise (BGR)
        kernel_size (tuple): ขนาดของ Kernel สำหรับ Gaussian Blur (กว้าง, สูง)

    Returns:
        numpy.ndarray: ภาพที่ลด noise แล้ว (BGR)
    """
    denoised_img = cv2.GaussianBlur(img_cv2, kernel_size, 0)
    return denoised_img


def remove_gaussian_noise_median_blur(img_cv2, kernel_size=5):
    """
    ลด Gaussian noise โดยใช้ Median Blur

    Args:
        img_cv2 (numpy.ndarray): ภาพที่มี Gaussian noise (BGR)
        kernel_size (int): ขนาดของ Kernel สำหรับ Median Blur (ต้องเป็นเลขคี่)

    Returns:
        numpy.ndarray: ภาพที่ลด noise แล้ว (BGR)
    """
    denoised_img = cv2.medianBlur(img_cv2, kernel_size)
    return denoised_img


# Function Cleansing
def cleansing_data(img_cv2, noise_stddev=20, blur_method="gaussian"):

    img_cv2_corrected = selective_color_correction(img_cv2.copy())

    img_cv2_balanced = gray_world_white_balance(img_cv2_corrected.copy())

    img_cv2_noisy = add_gaussian_noise(img_cv2_balanced.copy(), stddev=noise_stddev)

    if blur_method == "gaussian":
        img_cv2_denoised = remove_gaussian_noise_gaussian_blur(img_cv2_noisy.copy())
    elif blur_method == "median":
        img_cv2_denoised = remove_gaussian_noise_median_blur(img_cv2_noisy.copy())
    else:
        img_cv2_denoised = img_cv2_noisy.copy()
        print(f"Warning: Unknown blur_method '{blur_method}'. Using noisy image.")

    img_cv2_processed = preprocess_image(
        img_cv2_denoised.copy()
    )  # ทำสำเนาเพื่อไม่ให้กระทบภาพเดิม

    hsv_img = cv2.cvtColor(img_cv2_processed, cv2.COLOR_BGR2HSV)
    return hsv_img


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


def analyze_bag_color(image, box, confidence):
    """
    วิเคราะห์พื้นที่สีน้ำตาลและสีขาวขุ่นในถุง (โดยไม่ crop) และแสดงภาพที่มีการระบุพื้นที่

    Args:
        image_path (str): เส้นทางไปยังไฟล์รูปภาพ

    Returns:
        None
    """
    try:
        # 1. โหลดรูปภาพด้วย Pillow เพื่อวาด
        img_pil = image
        draw = ImageDraw.Draw(img_pil)

        # 2. โหลดรูปภาพด้วย OpenCV สำหรับการประมวลผลสี
        # img_cv2 = cv2.imread(image_path)
        hsv_img = cleansing_data(image)

        # 3. กำหนดขอบเขตของถุง (ปรับค่าตามความเหมาะสม)
        #    **คุณจะต้องปรับค่านี้ให้ตรงกับตำแหน่งของถุงในรูปภาพของคุณ**
        x1, y1, x2, y2 = box
        upper_bag_bbox = box  # ตัวอย่างค่า (ซ้าย, บน, ขวา, ล่าง)

        # 4. สร้าง Mask สำหรับถุง
        mask_bag = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = upper_bag_bbox
        # แปลงค่าเป็น integer
        x1_int = int(x1)
        y1_int = int(y1)
        x2_int = int(x2)
        y2_int = int(y2)
        cv2.rectangle(mask_bag, (x1_int, y1_int), (x2_int, y2_int), 255, -1)

        # 5. ตรวจจับสีน้ำตาล
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 255])
        mask_brown = cv2.inRange(hsv_img, lower_brown, upper_brown)
        mask_brown_in_bag = cv2.bitwise_and(mask_brown, mask_bag)
        brown_pixels = np.sum(mask_brown_in_bag > 0)

        # 6. ตรวจจับสีขาวขุ่น (อาจต้องปรับช่วงค่า HSV)
        lower_white_hsv = np.array([0, 0, 100])  # ปรับค่า Saturation และ Value
        upper_white_hsv = np.array([180, 80, 255])
        mask_white = cv2.inRange(hsv_img, lower_white_hsv, upper_white_hsv)
        mask_white_in_bag = cv2.bitwise_and(mask_white, mask_bag)
        white_pixels = np.sum(mask_white_in_bag > 0)

        # 7. หาพื้นที่ทั้งหมดในถุง
        bag_area = np.sum(mask_bag > 0)

        # 8. คำนวณเปอร์เซ็นต์
        # brown_percentage = (brown_pixels / bag_area) * 100 if bag_area > 0 else 0
        white_percentage = (white_pixels / bag_area) * 100 if bag_area > 0 else 0
        white_percentage = f"{white_percentage:.2f}"
        # print(f"พื้นที่สีน้ำตาลในถุง: {brown_percentage:.2f}%")
        # print(f"พื้นที่สีขาวขุ่นในถุง: {white_percentage:.2f}%")

        conf = confidence
        text = "white : " + str(white_percentage) + " - pot : " + str(conf)
        font = ImageFont.truetype("arial.ttf", 32)

        # 9. สร้างภาพที่มีการระบุพื้นที่
        masked_img_pil = img_pil.copy()
        masked_draw = ImageDraw.Draw(masked_img_pil)

        # # ระบายสีบริเวณสีน้ำตาล (สีแดง)
        # brown_coords = np.where(mask_brown_in_bag > 0)
        # for y, x in zip(brown_coords[0], brown_coords[1]):
        #     masked_draw.point((x, y), fill=(255, 0, 0))

        # ระบายสีบริเวณสีขาวขุ่น (สีน้ำเงิน)
        white_coords = np.where(mask_white_in_bag > 0)
        for y, x in zip(white_coords[0], white_coords[1]):
            masked_draw.point((x, y), fill=(0, 0, 255))

        # วาดกรอบรอบถุง
        masked_draw.rectangle(upper_bag_bbox, outline=(0, 255, 0), width=3)

        # ใส่ข้อความ ตัวหนา
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                masked_draw.text(
                    (x1_int + 40 + dx, y1_int - 40 + dy),
                    text,
                    font=font,
                    fill=(255, 0, 0),
                )

        # แสดงภาพ
        # img_pil.show("Original Image")
        # masked_img_pil.show("Detected Colors in Bag")

        # plt.imshow(masked_img_pil, cmap="gray")
        # plt.title("Show Image")
        # plt.axis("off")
        # plt.show()

        # โหลดและแปลงภาพเป็น base64
        # with open(masked_img_pil, 'rb') as image_file:
        #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # print(encoded_string)
        b64 = image_to_base64(masked_img_pil)
        # base64_str = image_np_to_base64(masked_img_pil)
        # print(b64)

        return jsonify({"base64": b64, "white": white_percentage})
        # return jsonify({'white':white_percentage})

    except FileNotFoundError:
        print("ไม่พบไฟล์รูปภาพ")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def predict_spawn(image):
    """
    ตัวอย่าง: ดึงข้อมูลจากฐานข้อมูลและพิมพ์ผลลัพธ์
    """
    now = datetime.datetime.now()
    print(f"Job executed at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        weights = "./v89_pot.pt"
        boxx, confidence = prediction(image, weights)
        data = analyze_bag_color(image, boxx, confidence)
        print(data)
        return data
    except Exception as e:
        print(f"error: {e}")
