from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from fastapi.responses import HTMLResponse



# Load the model once when the app starts
MODEL_DIR = "models/potato_disease_clf.keras"
model = tf.keras.models.load_model(MODEL_DIR)

# Define class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Initialize FastAPI
app = FastAPI(title="Potato Disease Classifier API")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)



def preprocess_image(image_bytes):
    """Preprocess the uploaded image for prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))  # Resize to match model input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Predict
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
