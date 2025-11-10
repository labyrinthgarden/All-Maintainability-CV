import tensorflow as tf
from tensorflow.keras.utils import plot_model

def create_model(num_classes):
    # Data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    # Base model: MobileNetV2 pretrained on ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model

if __name__=="__main__":
    num_classes=4
    model=create_model(num_classes)
    model.summary()
    plot_model(model,to_file="cnn_architecture.png",show_shapes=True,show_layer_names=True)
    print("Diagrama de la arquitectura guardado como cnn_architecture.png")
