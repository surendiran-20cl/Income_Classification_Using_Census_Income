import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the saved model
with open("lightgmb_income_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route("/")
def home():
    return "Income Classification Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get JSON data from request
        features = pd.DataFrame([data])  # Convert JSON to DataFrame
        prediction = model.predict(features)[0]  # Predict using the model
        return jsonify({"prediction": int(prediction)})  # Return result as JSON
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
