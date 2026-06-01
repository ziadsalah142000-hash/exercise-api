from flask import Flask, request, jsonify
from flask_sock import Sock
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
import cv2
import base64
import json
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

app = Flask(__name__)
sock = Sock(app)

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
BICEP_LMS  = [0, 11, 12, 14, 13, 16, 15, 23, 24]
SQUAT_LMS  = [0, 11, 12, 23, 24, 25, 26, 27, 28]
PLANK_LMS  = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

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

SQUAT_HEADERS = [
    "nose_x","nose_y","nose_z","nose_v",
    "left_shoulder_x","left_shoulder_y","left_shoulder_z","left_shoulder_v",
    "right_shoulder_x","right_shoulder_y","right_shoulder_z","right_shoulder_v",
    "left_hip_x","left_hip_y","left_hip_z","left_hip_v",
    "right_hip_x","right_hip_y","right_hip_z","right_hip_v",
    "left_knee_x","left_knee_y","left_knee_z","left_knee_v",
    "right_knee_x","right_knee_y","right_knee_z","right_knee_v",
    "left_ankle_x","left_ankle_y","left_ankle_z","left_ankle_v",
    "right_ankle_x","right_ankle_y","right_ankle_z","right_ankle_v",
]

PLANK_HEADERS = [
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
BICEP_LABELS = {"C": "Correct", "L": "Lean Back"}
SQUAT_LABELS = {"0": "Down", "1": "Up"}
PLANK_LABELS = {"0": "Correct", "1": "High Lower Back", "2": "Low Lower Back"}

EXERCISE_CONFIG = {
    "bicep": (BICEP_LMS, BICEP_HEADERS, bicep_scaler, bicep_model, BICEP_LABELS),
    "squat": (SQUAT_LMS, SQUAT_HEADERS, squat_scaler, squat_model, SQUAT_LABELS),
    "plank": (PLANK_LMS, PLANK_HEADERS, plank_scaler, plank_model, PLANK_LABELS),
}

# ==============================
# ADHD Helper
# ==============================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def detect_adhd_exercise(landmarks):
    def get_point(idx):
        lm = landmarks[idx]
        return [lm.x, lm.y]

    ls = get_point(11); rs = get_point(12)
    lh = get_point(23); rh = get_point(24)
    lk = get_point(25); rk = get_point(26)
    la = get_point(27); ra = get_point(28)
    lw = get_point(15); rw = get_point(16)

    lk_angle = calculate_angle(lh, lk, la)
    rk_angle = calculate_angle(rh, rk, ra)
    lh_angle = calculate_angle(ls, lh, lk)
    rh_angle = calculate_angle(rs, rh, rk)

    shoulder_y = (ls[1] + rs[1]) / 2
    wrist_y    = (lw[1] + rw[1]) / 2
    hip_y      = (lh[1] + rh[1]) / 2

    if (lk_angle < 90 or rk_angle < 90) and wrist_y < shoulder_y - 0.1:
        return "Tree Pose"
    if lh_angle < 80 and rh_angle < 80:
        return "Child's Pose"
    if lk_angle > 160 and rk_angle > 160 and wrist_y > hip_y:
        return "Deep Breathing"
    return "Keep Going!"

# ==============================
# Core processing
# ==============================
def process_image_bytes(image_bytes, lm_indices, headers, scaler, model, label_map, exercise_name):
    """Shared logic for both HTTP and WebSocket handlers."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return {"error": "No person detected"}

    landmarks = result.pose_landmarks[0]
    row = []
    for idx in lm_indices:
        lm = landmarks[idx]
        row += [lm.x, lm.y, lm.z, lm.visibility]

    # Draw skeleton
    annotated = image_array.copy()
    from mediapipe.framework.formats import landmark_pb2
    proto = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        proto.landmark.add(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
    mp_drawing.draw_landmarks(annotated, proto, mp_pose.POSE_CONNECTIONS)

    # Predict
    X = pd.DataFrame([row], columns=headers)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]

    # Encode annotated image
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', annotated_bgr)
    image_b64 = base64.b64encode(buffer).decode('utf-8')

    landmarks_list = [{"id": i, "x": lm.x, "y": lm.y, "z": lm.z} for i, lm in enumerate(landmarks)]

    return {
        "exercise": exercise_name,
        "prediction": str(pred),
        "message": label_map.get(str(pred), "Unknown"),
        "image": image_b64,
        "landmarks": landmarks_list
    }


def process_adhd_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return {"error": "No person detected"}

    landmarks = result.pose_landmarks[0]
    exercise  = detect_adhd_exercise(landmarks)
    lm_list   = [{"id": i, "x": lm.x, "y": lm.y, "z": lm.z} for i, lm in enumerate(landmarks)]

    return {"exercise": "adhd", "message": exercise, "landmarks": lm_list}


# ==============================
# HTTP Routes (kept as-is)
# ==============================
@app.route("/")
def home():
    return (
        "Exercise Pose Detection API 💪 | "
        "HTTP: /predict/bicep | /predict/squat | /predict/plank | /predict/adhd | "
        "WebSocket: /ws/bicep | /ws/squat | /ws/plank | /ws/adhd"
    )

def http_predict(exercise_name):
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})
        image_bytes = request.files["image"].read()
        if exercise_name == "adhd":
            return jsonify(process_adhd_bytes(image_bytes))
        lm_indices, headers, scaler, model, label_map = EXERCISE_CONFIG[exercise_name]
        return jsonify(process_image_bytes(image_bytes, lm_indices, headers, scaler, model, label_map, exercise_name))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict/bicep", methods=["POST"])
def predict_bicep(): return http_predict("bicep")

@app.route("/predict/squat", methods=["POST"])
def predict_squat(): return http_predict("squat")

@app.route("/predict/plank", methods=["POST"])
def predict_plank(): return http_predict("plank")

@app.route("/predict/adhd", methods=["POST"])
def predict_adhd(): return http_predict("adhd")


# ==============================
# WebSocket Routes
# ==============================
# Flutter sends frames as JSON: { "frame": "<base64 jpeg>" }
# Server replies with JSON:     { "exercise": "...", "prediction": "...", "message": "...", "landmarks": [...] }
# Note: annotated image is omitted from WS responses to keep latency low.
#       Flutter renders its own overlay using the landmarks array.

def ws_predict(ws, exercise_name):
    """Generic WebSocket handler for all exercise types."""
    while True:
        try:
            data = ws.receive()
            if data is None:
                break

            # Accept raw bytes OR JSON { "frame": "<base64>" }
            if isinstance(data, bytes):
                image_bytes = data
            else:
                payload = json.loads(data)
                image_bytes = base64.b64decode(payload["frame"])

            if exercise_name == "adhd":
                result = process_adhd_bytes(image_bytes)
            else:
                lm_indices, headers, scaler, model, label_map = EXERCISE_CONFIG[exercise_name]
                result = process_image_bytes(
                    image_bytes, lm_indices, headers, scaler, model, label_map, exercise_name
                )
                # Drop the heavy base64 image — Flutter renders its own overlay
                result.pop("image", None)

            ws.send(json.dumps(result))

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))
            break


@sock.route("/ws/bicep")
def ws_bicep(ws): ws_predict(ws, "bicep")

@sock.route("/ws/squat")
def ws_squat(ws): ws_predict(ws, "squat")

@sock.route("/ws/plank")
def ws_plank(ws): ws_predict(ws, "plank")

@sock.route("/ws/adhd")
def ws_adhd(ws): ws_predict(ws, "adhd")


# ==============================
# Run
# ==============================
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
