from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS
import joblib

# =========================
# Gemini AI
# =========================
import google.generativeai as genai

# =========================
# Load ML Models (UPDATED)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Eye Vision Risk Model
with open(os.path.join(BASE_DIR, "eye_vision_risk_model.pkl"), "rb") as f:
    eye_model = pickle.load(f)

# Diabetes Model (SCALER REMOVED)
diabetes_model = joblib.load(
    os.path.join(BASE_DIR, "diabetes_prediction_model.pkl")
)

# =========================
# Gemini Configuration
# =========================
genai.configure(
    api_key=os.environ.get("GEMINI_API_KEY") or "AIzaSyDmnXmATVPgIEsf6wn9QnJUgux4n__G0fk"
)

# Correct Gemini model
gemini_model = genai.GenerativeModel("gemini-1.0-pro")

# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "MediVision Backend is running"

# =========================
# Eye Risk Prediction
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
# Diabetes Prediction (FIXED)
# =========================
@app.route("/predict-diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.json

        features = np.array([[
            float(data["age"]),
            float(data["bmi"]),
            float(data["bp"]),
            float(data["glucose"])
        ]])

        prediction = diabetes_model.predict(features)[0]

        return jsonify({
            "prediction": int(prediction),
            "risk_level": "High Risk" if prediction == 1 else "Low Risk"
        })

    except Exception as e:
        print("Diabetes Error:", e)
        return jsonify({"error": str(e)}), 400

# =========================
# Health Assistant
# =========================
@app.route("/health-assistant", methods=["POST"])
def health_assistant():
    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please ask a health-related question."})

        prompt = f"""
You are MediVision Health Assistant.

Rules:
- Do NOT diagnose diseases
- Do NOT prescribe medicines
- Provide general health & lifestyle advice only
- Always suggest consulting a doctor for serious issues

User question:
{user_message}
"""

        response = gemini_model.generate_content(prompt)

        return jsonify({
            "reply": response.text
        })

    except Exception as e:
        print("Gemini Error:", e)
        return jsonify({
            "reply": "Health assistant is temporarily unavailable."
        }), 500

# =========================
# Run Server
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
