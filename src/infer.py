import tensorflow as tf
import numpy as np
import cv2

MODEL_DIR = "../models/saved_model"
IMG_PATH = "../data/raw/test.jpg"

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_DIR)
    class_names = ["ceilingGood", "ceilingDamaged"]

    img = cv2.imread(IMG_PATH)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    print(f"🔎 Predicción: {pred_class} ({np.max(preds)*100:.2f}%)")

