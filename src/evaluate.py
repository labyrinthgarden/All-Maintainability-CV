import tensorflow as tf
from dataset import load_dataset

DATA_DIR = "data/raw"
MODEL_DIR = "models/saved_model.keras"

if __name__ == "__main__":
    val_ds = load_dataset(DATA_DIR)
    model = tf.keras.models.load_model(MODEL_DIR)

    loss, acc = model.evaluate(val_ds)
    print(f"-> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
