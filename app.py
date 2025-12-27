from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS


# Load model
with open("eye_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Eye Risk Prediction API is running"

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

    prediction = model.predict([features])[0]

    result_map = {
        0: "Normal Vision",
        1: "Mild Vision Risk",
        2: "High Vision Risk"
    }

    return jsonify({
        "risk_level": int(prediction),
        "result": result_map[prediction]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
