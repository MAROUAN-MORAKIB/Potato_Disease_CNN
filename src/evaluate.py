import tensorflow as tf
import json, os
import mlflow
import mlflow.tensorflow

IMG_SIZE = 256
BATCH_SIZE = 32

# Load prepared test set
test_ds = tf.data.experimental.load("data/processed/test")
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Load trained model
model = tf.keras.models.load_model("models/potato_cnn.h5")

# Evaluate
loss, acc = model.evaluate(test_ds)

# === MLflow Tracking ===
mlflow.set_experiment("Potato_Disease_CNN")

# Start a nested run (inside the training run)
with mlflow.start_run(nested=True):
    mlflow.log_metric("test_accuracy", float(acc))
    mlflow.log_metric("test_loss", float(loss))

    # Optionally log confusion matrix / plots later
    # e.g. mlflow.log_artifact("confusion_matrix.png")

    # Save evaluation metrics for DVC
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval.json", "w") as f:
        json.dump({
            "test_loss": float(loss),
            "test_accuracy": float(acc)
        }, f)

print(f"✅ Evaluation done: test_accuracy={acc:.4f}, test_loss={loss:.4f}")
