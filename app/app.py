import os
from io import BytesIO

import numpy as np
from flask import Flask, jsonify, request, render_template
from PIL import Image
from tensorflow.keras.models import load_model

from labels import CLASS_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local path
local_model = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "plant_disease_model.h5")
)

# Docker path
docker_model = "/app/model/plant_disease_model.h5"

MODEL_PATH = docker_model if os.path.exists(docker_model) else local_model

print("BASE_DIR =", BASE_DIR)
print("MODEL_PATH =", MODEL_PATH)

app = Flask(__name__)
model = load_model(MODEL_PATH)

IMAGE_SIZE = (64, 64)


@app.route("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "message": "Plant Disease Detection API is running"
    })


def preprocess_image(file_storage):
    image_bytes = file_storage.read()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    return np.expand_dims(image_array, axis=0)


@app.post("/predict")
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image file provided. Use form field 'image'."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:
        processed_image = preprocess_image(image_file)

        predictions = model.predict(processed_image)

        scores = predictions[0]

        predicted_index = int(np.argmax(scores))

        confidence = float(scores[predicted_index])

        return jsonify({
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )