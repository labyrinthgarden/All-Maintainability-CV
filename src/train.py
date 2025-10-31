import tensorflow as tf
from dataset import load_dataset
from model import create_model
import os

DATA_DIR = "data/raw"
MODEL_PATH = "models/saved_model.keras"

if __name__ == "__main__":
    # Get both the dataset and class names in one call
    train_ds = load_dataset(DATA_DIR)
    class_names = load_dataset(DATA_DIR)
    num_classes = len(class_names)

    model = create_model(num_classes)
    model.fit(train_ds, epochs=3)

    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"-->> Modelo guardado en {MODEL_PATH}")
