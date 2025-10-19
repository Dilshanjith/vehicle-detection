import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import sys
from config import IMAGE_SIZE, MODEL_SAVE_PATH, CLASS_NAMES

def load_and_predict_image(image_path, model_path=MODEL_SAVE_PATH):
    """Loads a saved model and predicts a single image."""

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}. Please train the model first.")
        return

    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found at {image_path}")
        return

    # Load model
    print(f"🔄 Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")

    # Preprocess image
    print(f"🖼️ Loading and preprocessing image: {image_path}...")
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    print("🤖 Making prediction...")
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    predicted_label = CLASS_NAMES[predicted_index] if CLASS_NAMES else f"Class {predicted_index}"

    # Display result
    print("\n--- Prediction Result ---")
    print(f"Image Path: {image_path}")
    print(f"Predicted Class: {predicted_label.capitalize()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("-------------------------\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"path_to_image.jpg\"")
    else:
        image_path = sys.argv[1].strip('"')
        load_and_predict_image(image_path)
