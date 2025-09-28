import tensorflow as tf
from keras import layers, models
import json, os
import mlflow
import mlflow.tensorflow
from keras.callbacks import Callback

IMG_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
N_CLASSES = 3

# Load prepared datasets
train_ds = tf.data.experimental.load("data/processed/train")
val_ds = tf.data.experimental.load("data/processed/val")

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds   = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2)
])

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(N_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === MLflow Tracking ===
mlflow.set_experiment("Potato_Disease_CNN")


class MLflowLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mlflow.log_metric("train_accuracy", logs.get("accuracy"), step=epoch)
        mlflow.log_metric("val_accuracy", logs.get("val_accuracy"), step=epoch)
        mlflow.log_metric("train_loss", logs.get("loss"), step=epoch)
        mlflow.log_metric("val_loss", logs.get("val_loss"), step=epoch)

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("img_size", IMG_SIZE)

    # Train with MLflow callback
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[MLflowLogger()]
    )

    # Save model locally
    os.makedirs("models", exist_ok=True)
    model.save("models/potato_cnn.h5")

    # Log model to MLflow
    mlflow.tensorflow.log_model(model, artifact_path="model")

    # Save last metrics for DVC
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train.json", "w") as f:
        json.dump({
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1])
        }, f)

print("✅ Training finished, metrics logged live to MLflow + DVC")
