import tensorflow as tf
import pathlib

def load_dataset(data_dir, img_size=(224, 224), batch_size=16):
    data_dir = pathlib.Path(data_dir)
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

