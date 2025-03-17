from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/dental_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"cavity_detected": bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
