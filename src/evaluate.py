import tensorflow as tf
import json, os

IMG_SIZE = 256
BATCH_SIZE = 32

# Load prepared test set
test_ds = tf.data.experimental.load("data/processed/test")
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Load trained model
model = tf.keras.models.load_model("models/potato_cnn.h5")

# Evaluate
loss, acc = model.evaluate(test_ds)

# Save evaluation metrics
os.makedirs("metrics", exist_ok=True)
with open("metrics/eval.json", "w") as f:
    json.dump({
        "test_loss": float(loss),
        "test_accuracy": float(acc)
    }, f)

print(f"✅ Evaluation done: accuracy={acc:.4f}, loss={loss:.4f}")
