from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("trash_classifier_real.h5")
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (64, 64)) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = preprocess_image(file)
    
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({"class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
