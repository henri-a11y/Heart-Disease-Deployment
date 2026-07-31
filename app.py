"""
Flask web application for heart disease prediction.
Serves a Bootstrap UI and a REST API at POST /predict.
"""

import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
model_name = artifact["model_name"]
model_accuracy = artifact["accuracy"]
feature_columns = artifact["features"]


def parse_patient_data(data: dict) -> np.ndarray:
    """Extract and validate feature values from input data."""
    values = []
    for feature in feature_columns:
        if feature not in data:
            raise ValueError(f"Missing required field: {feature}")
        values.append(float(data[feature]))
    return np.array([values])


def make_prediction(features: np.ndarray) -> dict:
    """Run prediction and return formatted result."""
    prediction = int(model.predict(features)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
        probability = 1 / (1 + np.exp(-score))

    has_disease = prediction == 1
    return {
        "prediction": prediction,
        "has_heart_disease": has_disease,
        "result": "Heart Disease Detected" if has_disease else "No Heart Disease",
        "probability": round(probability, 4) if probability is not None else None,
        "model_used": model_name,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the prediction form and handle form submissions."""
    prediction_result = None
    error = None

    if request.method == "POST":
        try:
            data = {key: request.form[key] for key in feature_columns}
            features = parse_patient_data(data)
            prediction_result = make_prediction(features)
        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        features=feature_columns,
        prediction=prediction_result,
        error=error,
        model_name=model_name,
        model_accuracy=model_accuracy,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """REST API endpoint for heart disease prediction."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    try:
        features = parse_patient_data(data)
        result = make_prediction(features)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy", "model": model_name}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
