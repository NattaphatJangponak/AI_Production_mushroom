import os

# ตั้งค่า environment variables ก่อน import libraries อื่น
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from predict_spawn import predict_spawn
from predict_growth import predict_growth
from predict_spawn_seg import predict_spawn_seg
from predict_growth_seg import predict_growth_seg

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


def clean_base64(b64_str):
    if b64_str:
        base64_str = b64_str.replace("\n", "").replace("\r", "").replace(" ", "")
        missing_padding = len(base64_str) % 4
        if missing_padding:
            base64_str += "=" * (4 - missing_padding)

        return base64_str
    else:
        return False


def spawn_predict(base64_string):
    """แปลงสตริง Base64 เป็นอ็อบเจ็กต์ Image ของ PIL."""
    try:
        base64_string = clean_base64(base64_string)
        if base64_string:
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_bytes))
            data = predict_spawn(image)
            return data
        else:
            return False
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแปลง Base64: {e}")
        return jsonify({"error growth": str(e)}), 500


@app.route("/spawn", methods=["POST"])
@cross_origin()
def spawn():
    try:
        data = request.json
        base64_str = data["base64"]
        data = spawn_predict(base64_str)
        return data
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route("/spawn_seg", methods=["POST"])
# @cross_origin()
# def spawn_seg():
#     try:
#         data = request.json
#         base64_str = data["base64"]
#         base64_str = clean_base64(base64_str)
#         image_bytes = base64.b64decode(base64_str)
#         image = Image.open(BytesIO(image_bytes))
#         return predict_spawn_seg(image)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route("/spawn_seg", methods=["POST"])
@cross_origin()
def spawn_seg():
    raw = request.json
    status = 0
    ans = []
    for i in raw:
        try:
            data = i
            print(f"Processing ID: {data['id']}")
            print(f"Base64 Length: {len(data['base64'])}")
            base64_str = data["base64"]
            base64_str = clean_base64(base64_str)
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_bytes))
            a2 = predict_spawn_seg(image)

            ans.append({"id": f"{i['id']}", "base64": a2[0], "white": a2[1]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    # print(f"Spawn Segmentation Results: {ans}")
    return jsonify(ans), 200


def growth_predict(base64_string):
    """แปลงสตริง Base64 เป็นอ็อบเจ็กต์ Image ของ PIL."""
    try:

        base64_string = clean_base64(base64_string)
        if base64_string:
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_bytes))
            data = predict_growth(image)
            return data
        else:
            return False
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแปลง Base64: {e}")
        return jsonify({"error Base64": str(e)}), 500


@app.route("/growth", methods=["POST"])
@cross_origin()
def growth():
    try:
        data = request.json
        base64_str = data["base64"]
        data = growth_predict(base64_str)
        return data
    except Exception as e:
        return jsonify({"error growth": str(e)}), 500


@app.route("/growth_seg", methods=["POST"])
@cross_origin()
def growth_seg():
    raw = request.json
    ans = []

    # บังคับให้เป็น array ของรายการเท่านั้น (รูปแบบเดียวกับ /spawn_seg)
    if not isinstance(raw, list):
        return jsonify({"error": "Invalid payload: must be an array of objects"}), 400

    for idx, item in enumerate(raw):
        try:
            if not isinstance(item, dict):
                return (
                    jsonify(
                        {"error": f"Invalid item at index {idx}: must be an object"}
                    ),
                    400,
                )

            item_id = str(item.get("id", idx + 1))
            base64_val = item.get("base64")
            if not base64_val:
                return jsonify({"error": f"Missing 'base64' for id {item_id}"}), 400

            print(f"Processing ID: {item_id}")
            print(f"Base64 Length: {len(base64_val)}")

            base64_str = clean_base64(base64_val)
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_bytes))
            a3 = predict_growth_seg(image)

            ans.append({"id": item_id, "base64": a3[0], "status": a3[1]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"Growth Segmentation Results: {ans}")
    return jsonify(ans), 200


@app.route("/")
@cross_origin()
def helloWorld():
    return "V89's test flask"


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8989, debug=True, use_reloader=False)
