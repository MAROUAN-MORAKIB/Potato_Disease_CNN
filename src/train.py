import tensorflow as tf
from keras import layers, models
import json, os

IMG_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
N_CLASSES = 3

# Load prepared datasets
train_ds = tf.data.experimental.load("data/processed/train")
val_ds = tf.data.experimental.load("data/processed/val")

# Optimize performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Model
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

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, batch_size=BATCH_SIZE)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/potato_cnn.h5")

# Save training metrics
os.makedirs("metrics", exist_ok=True)
with open("metrics/train.json", "w") as f:
    json.dump({
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1])
    }, f)

print("✅ Model trained and saved at models/potato_cnn.h5")
