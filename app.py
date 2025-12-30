from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS
import joblib

# =========================
# Load Models
# =========================

# Eye Risk Model (pickle)
with open("eye_risk_model.pkl", "rb") as f:
    eye_model = pickle.load(f)

# Diabetes Model + Scaler (joblib)
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("diabetes_scaler.pkl")

# =========================
# App Setup
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return "MediVision Backend is running"

# =========================
# Eye Risk Prediction (UNCHANGED)
# =========================
@app.route("/predict-eye-risk", methods=["POST"])
def predict_eye_risk():
    data = request.json

    features = [
        data["min_font_size"],
        data["error_count"],
        data["screen_time"],
        data["eye_strain"],
        data["headache"],
        data["distance_cm"]
    ]

    prediction = eye_model.predict([features])[0]

    result_map = {
        0: "Normal Vision",
        1: "Mild Vision Risk",
        2: "High Vision Risk"
    }

    return jsonify({
        "risk_level": int(prediction),
        "result": result_map[prediction]
    })

# =========================
# Diabetes Prediction (NEW – CORRECT WAY)
# =========================
@app.route("/predict-diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.json

        # Input order MUST match training
        features = np.array([[
            float(data["age"]),
            float(data["bmi"]),
            float(data["bp"]),
            float(data["glucose"])
        ]])

        # Scale input
        features_scaled = diabetes_scaler.transform(features)

        # Predict
        prediction = diabetes_model.predict(features_scaled)[0]

        risk = "High Risk" if prediction == 1 else "Low Risk"

        return jsonify({
            "prediction": int(prediction),
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =========================
# Run Server
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
