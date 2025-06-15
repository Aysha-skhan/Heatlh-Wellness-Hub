from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Occupation encoding mapping
OCCUPATION_MAPPING = {
    'Student': 0,
    'Engineer': 1,
    'Healthcare': 2,
    'Teacher': 3,
    'Sales Representative': 4,
    'Software Engineer': 5,
    'Doctor': 6,
    'Other': 7
}

# Try to load existing model and scaler, or create new ones
try:
    model = joblib.load("lifestyle_model.pkl")
    print("Loaded existing model")
except:
    print("Creating new model")
    # Create a simple dummy model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train with dummy data
    X_dummy = np.random.rand(100, 11)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, "lifestyle_model.pkl")

try:
    scaler = joblib.load("scaler.pkl")
    print("Loaded existing scaler")
except:
    print("Creating new scaler")
    scaler = StandardScaler()
    X_dummy = np.random.rand(100, 11)
    scaler.fit(X_dummy)
    joblib.dump(scaler, "scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predictor')
def predictor():
    return render_template("predictor.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        # Feature processing
        features = [
            1 if data.get("gender") == "Female" else 0,
            int(data.get("age", 0)),
            OCCUPATION_MAPPING.get(data.get("occupation", ""), 0),
            float(data.get("sleep_duration", 0)),
            int(data.get("quality_sleep", 5)),
            int(data.get("physical_activity", 0)),
            int(data.get("stress_level", 5)),
            int(data.get("heart_rate", 72)),
            int(data.get("daily_steps", 0)),
            int(data.get("systolic", 115)),
            int(data.get("diastolic", 75))
        ]

        input_data = np.array(features).reshape(1, -1)
        
        if 'scaler' in globals():
            input_data = scaler.transform(input_data)

        prediction = int(model.predict(input_data)[0])
        prediction_proba = model.predict_proba(input_data)[0].tolist()

        return jsonify({
            "prediction": prediction,
            "probabilities": prediction_proba,
            "confidence": float(prediction_proba[prediction])
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
