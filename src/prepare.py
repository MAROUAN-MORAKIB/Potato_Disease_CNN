import tensorflow as tf
import os

IMG_SIZE = 256
BATCH_SIZE = 32

dataset_dir = "data/PlantVillage"

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    shuffle=True,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

def splitting_data(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = splitting_data(dataset)

# Save splits to disk (TensorFlow format)
tf.data.experimental.save(train_ds, "data/processed/train")
tf.data.experimental.save(val_ds, "data/processed/val")
tf.data.experimental.save(test_ds, "data/processed/test")

print("✅ Dataset prepared and saved at data/processed/")
