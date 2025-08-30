import tensorflow as tf
from dataset import load_dataset
from model import create_model
import os

DATA_DIR = "../data/raw"
MODEL_DIR = "../models/saved_model"

if __name__ == "__main__":
    train_ds = load_dataset(DATA_DIR)
    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = create_model(num_classes)
    model.fit(train_ds, epochs=3)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_DIR)
    print(f"-->> Modelo guardado en {MODEL_DIR}")

