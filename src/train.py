import tensorflow as tf
from dataset import load_dataset
from model import create_model
import os

DATA_DIR = "data/raw"
MODEL_PATH = "models/saved_model.keras"

if __name__ == "__main__":
    import pathlib
    # Usar validation_split para separar entrenamiento y validación
    data_dir = pathlib.Path(DATA_DIR)
    img_size = (224, 224)
    batch_size = 16
    seed = 123

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)

    model = create_model(num_classes)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20
    )

    # Mostrar accuracy y loss de validación final
    print("Accuracy final de entrenamiento:", history.history["accuracy"][-1])
    print("Accuracy final de validación:", history.history["val_accuracy"][-1])
    print("Loss final de entrenamiento:", history.history["loss"][-1])
    print("Loss final de validación:", history.history["val_loss"][-1])

    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"-->> Modelo guardado en {MODEL_PATH}")
