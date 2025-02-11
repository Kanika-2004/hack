from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("startup_funding_model.h5")  # Ensure your model is saved in this path

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)  # Ensure input is in correct shape

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'funding_prediction': float(prediction[0][0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
