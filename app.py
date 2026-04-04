from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import mediapipe as mp
from PIL import Image
import io

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
# MediaPipe Setup
# ==============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ==============================
# Extract Landmarks
# ==============================
def extract_landmarks(image):
    results = pose.process(image)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    IMPORTANT_LMS = [0,11,12,14,13,16,15,23,24]

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

# ----------- Image API -----------
@app.route("/predict-image", methods=["POST"])
def predict_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})

        file = request.files["image"]

        # قراءة الصورة باستخدام PIL بدل cv2
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

        label_map = {
            "C": "Correct",
            "L": "Lean Back"
        }

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
