import tensorflow as tf
import numpy as np
import cv2

MODEL_DIR = "models/saved_model.keras"
IMG_PATH = "/home/aphr/Documents/nada/2.jpeg"

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_DIR)
    class_names = ["paredes_1", "paredes_2", "ceiling_1", "ceiling_2"]

    img = cv2.imread(IMG_PATH)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    preds = model.predict(img_array)

    pred_class = class_names[np.argmax(preds)]
    print(f"🔎 Predicción: {pred_class} ({np.max(preds)*100:.2f}%)")
