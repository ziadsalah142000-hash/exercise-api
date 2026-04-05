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
# Download pose model if needed
# ==============================
MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        MODEL_PATH
    )
    print("Pose model downloaded!")

# ==============================
# MediaPipe Setup
# ==============================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# ==============================
# Load Models
# ==============================
with open("models/bicep_model/input_scaler.pkl", "rb") as f:
    bicep_scaler = pickle.load(f)
with open("models/bicep_model/KNN_model.pkl", "rb") as f:
    bicep_model = pickle.load(f)

with open("models/squat_model/input_scaler.pkl", "rb") as f:
    squat_scaler = pickle.load(f)
with open("models/squat_model/LR_model.pkl", "rb") as f:
    squat_model = pickle.load(f)

with open("models/plank_model/input_scaler.pkl", "rb") as f:
    plank_scaler = pickle.load(f)
with open("models/plank_model/LR_model.pkl", "rb") as f:
    plank_model = pickle.load(f)

# ==============================
# Landmarks Indices
# ==============================
# Bicep: nose, shoulders, elbows, wrists, hips
BICEP_LMS = [0, 11, 12, 14, 13, 16, 15, 23, 24]

# Squat & Plank: nose, shoulders, elbows, wrists, hips, knees, ankles, heels, foot index
FULL_LMS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# ==============================
# Headers
# ==============================
BICEP_HEADERS = [
    "nose_x","nose_y","nose_z","nose_v",
    "left_shoulder_x","left_shoulder_y","left_shoulder_z","left_shoulder_v",
    "right_shoulder_x","right_shoulder_y","right_shoulder_z","right_shoulder_v",
    "right_elbow_x","right_elbow_y","right_elbow_z","right_elbow_v",
    "left_elbow_x","left_elbow_y","left_elbow_z","left_elbow_v",
    "right_wrist_x","right_wrist_y","right_wrist_z","right_wrist_v",
    "left_wrist_x","left_wrist_y","left_wrist_z","left_wrist_v",
    "left_hip_x","left_hip_y","left_hip_z","left_hip_v",
    "right_hip_x","right_hip_y","right_hip_z","right_hip_v",
]

FULL_HEADERS = [
    "nose_x","nose_y","nose_z","nose_v",
    "left_shoulder_x","left_shoulder_y","left_shoulder_z","left_shoulder_v",
    "right_shoulder_x","right_shoulder_y","right_shoulder_z","right_shoulder_v",
    "left_elbow_x","left_elbow_y","left_elbow_z","left_elbow_v",
    "right_elbow_x","right_elbow_y","right_elbow_z","right_elbow_v",
    "left_wrist_x","left_wrist_y","left_wrist_z","left_wrist_v",
    "right_wrist_x","right_wrist_y","right_wrist_z","right_wrist_v",
    "left_hip_x","left_hip_y","left_hip_z","left_hip_v",
    "right_hip_x","right_hip_y","right_hip_z","right_hip_v",
    "left_knee_x","left_knee_y","left_knee_z","left_knee_v",
    "right_knee_x","right_knee_y","right_knee_z","right_knee_v",
    "left_ankle_x","left_ankle_y","left_ankle_z","left_ankle_v",
    "right_ankle_x","right_ankle_y","right_ankle_z","right_ankle_v",
    "left_heel_x","left_heel_y","left_heel_z","left_heel_v",
    "right_heel_x","right_heel_y","right_heel_z","right_heel_v",
    "left_foot_index_x","left_foot_index_y","left_foot_index_z","left_foot_index_v",
    "right_foot_index_x","right_foot_index_y","right_foot_index_z","right_foot_index_v",
]

# ==============================
# Label Maps
# ==============================
BICEP_LABELS  = {"C": "Correct", "L": "Lean Back"}
SQUAT_LABELS  = {"C": "Correct", "L": "Lean Back", "H": "High Back"}
PLANK_LABELS  = {"C": "Correct", "H": "High Lower Back", "L": "Low Lower Back"}

# ==============================
# Helper
# ==============================
def extract_landmarks(image_array, lm_indices):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    result = detector.detect(mp_image)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    row = []
    for idx in lm_indices:
        lm = landmarks[idx]
        row += [lm.x, lm.y, lm.z, lm.visibility]
    return row


def predict_exercise(exercise, lm_indices, headers, scaler, model, label_map):
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})

        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = np.array(image)

        landmarks = extract_landmarks(image, lm_indices)
        if landmarks is None:
            return jsonify({"error": "No person detected"})

        X = pd.DataFrame([landmarks], columns=headers)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        return jsonify({
            "exercise": exercise,
            "prediction": str(pred),
            "message": label_map.get(str(pred), "Unknown")
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return "Exercise Pose Detection API 💪 | Endpoints: /predict/bicep | /predict/squat | /predict/plank"


@app.route("/predict/bicep", methods=["POST"])
def predict_bicep():
    return predict_exercise("bicep", BICEP_LMS, BICEP_HEADERS, bicep_scaler, bicep_model, BICEP_LABELS)


@app.route("/predict/squat", methods=["POST"])
def predict_squat():
    return predict_exercise("squat", FULL_LMS, FULL_HEADERS, squat_scaler, squat_model, SQUAT_LABELS)


@app.route("/predict/plank", methods=["POST"])
def predict_plank():
    return predict_exercise("plank", FULL_LMS, FULL_HEADERS, plank_scaler, plank_model, PLANK_LABELS)


# ==============================
# Run
# ==============================
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
