from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import io
import urllib.request

app = Flask(__name__)

# ==============================
# Load Models
# ==============================
with open("models/input_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/KNN_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==============================
# Headers (features)
# ==============================
HEADERS = [
    "nose_x","nose_y","nose_z","nose_v",
    "left_shoulder_x","left_shoulder_y","left_shoulder_z","left_shoulder_v",
    "right_shoulder_x","right_shoulder_y","right_shoulder_z","right_shoulder_v",
    "right_elbow_x","right_elbow_y","right_elbow_z","right_elbow_v",
    "left_elbow_x","left_elbow_y","left_elbow_z","left_elbow_v",
    "right_wrist_x","right_wrist_y","right_wrist_z","right_wrist_v",
    "left_wrist_x","left_wrist_y","left_wrist_z","left_wrist_v",
    "left_hip_x","left_hip_y","left_hip_z","left_hip_v",
    "right_hip_x","right_hip_y","right_hip_z","right_hip_v"
]

# ==============================
# Download pose model if needed
# ==============================
MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        MODEL_PATH
    )

# ==============================
# MediaPipe Setup (New API)
# ==============================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# ==============================
# Extract Landmarks
# ==============================
IMPORTANT_LMS = [0, 11, 12, 14, 13, 16, 15, 23, 24]

def extract_landmarks(image_array):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    result = detector.detect(mp_image)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    row = []
    for idx in IMPORTANT_LMS:
        lm = landmarks[idx]
        row += [lm.x, lm.y, lm.z, lm.visibility]
    return row

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return "Bicep Image API Running 🔥"

@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = np.array(image)
        landmarks = extract_landmarks(image)
        if landmarks is None:
            return jsonify({"error": "No person detected"})
        if len(landmarks) != len(HEADERS):
            return jsonify({"error": "Landmarks size mismatch"})
        X = pd.DataFrame([landmarks], columns=HEADERS)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        label_map = {"C": "Correct", "L": "Lean Back"}
        return jsonify({
            "prediction": str(pred),
            "message": label_map.get(pred, "Unknown")
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# Run App
# ==============================
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
