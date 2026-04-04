from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# load models
with open("models/input_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/KNN_model.pkl", "rb") as f:
    model = pickle.load(f)

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

@app.route("/")
def home():
    return "Bicep API Running 💪"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("landmarks")

        if data is None:
            return jsonify({"error": "No landmarks provided"})

        if len(data) != len(HEADERS):
            return jsonify({
                "error": f"Expected {len(HEADERS)} values, got {len(data)}"
            })

        X = pd.DataFrame([data], columns=HEADERS)
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

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
