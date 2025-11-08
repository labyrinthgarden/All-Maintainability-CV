import tensorflow as tf
from tensorflow.keras.utils import plot_model

def create_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
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
