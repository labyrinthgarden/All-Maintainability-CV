import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import cv2
import tensorflow as tf

import sys
from pathlib import Path
import shutil
from datetime import datetime
from typing import List

# Add src to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataset import load_dataset
from model import create_model

DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "raw")
MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "saved_model.keras")
CLASS_NAMES = ["paredes_exteriores_agrietadas", "paredes_exteriores_buen_estado", "ceilingDamaged", "ceilingGood"]
TRAIN_DATA_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "training_images")



def ensure_model():
    """Train and save the model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        train_ds = load_dataset(DATA_DIR)
        # Get class names from dataset
        class_names = train_ds.class_names if hasattr(train_ds, "class_names") else CLASS_NAMES
        num_classes = len(class_names)
        model = create_model(num_classes)
        model.fit(train_ds, epochs=3)
        model.save(MODEL_PATH)
        print(f"Model trained")

def load_model():
    ensure_model()
    return tf.keras.models.load_model(MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_model()
    yield

app = FastAPI(
    title="All-Maintainability-CV API",
    description="API for training, inference, and serving results for the maintainability CV model.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/infer/")
async def infer_images(files: List[UploadFile] = File(...)):
    """Receive multiple images and return predictions for each."""
    try:
        model = load_model()
        results = []
        for file in files:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                results.append({
                    "filename": file.filename,
                    "error": "Could not decode image"
                })
                continue
            img_resized = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0) / 255.0
            preds = model.predict(img_array)
            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(np.max(preds))
            results.append({
                "filename": file.filename,
                "predicted_class": pred_class,
                "confidence": confidence,
                "class_names": CLASS_NAMES
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")

@app.post("/upload-training-data/")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    label: str = Form(...)
):
    """
    Receive images and a label, store them in the training images directory with unique names.
    """
    try:
        label_dir = os.path.join(TRAIN_DATA_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        for file in files:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            new_filename = f"user_{timestamp}_{file.filename}"
            file_path = os.path.join(label_dir, new_filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        return {"message": "Training images uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload training data: {str(e)}")

@app.post("/train/")
def train_model():
    """Trigger model training (overwrites existing model)."""
    try:
        train_ds = load_dataset(DATA_DIR)
        class_names = train_ds.class_names if hasattr(train_ds, "class_names") else CLASS_NAMES
        num_classes = len(class_names)
        model = create_model(num_classes)
        model.fit(train_ds, epochs=3)
        model.save(MODEL_PATH)
        return {"message": f"Model retrained"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/status/")
def status():
    """Check if the model exists and is ready."""
    exists = os.path.exists(MODEL_PATH)
    return {"model_exists": exists, "model_path": MODEL_PATH}

@app.get("/")
def root():
    return {"message": "All-Maintainability-CV API is running."}
