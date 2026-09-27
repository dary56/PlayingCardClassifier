import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from model import load_trained_model
from dataset import get_transform
from config import CHECKPOINT_PATH, CLASSES_PATH


app = FastAPI(title="Playing Card Classifier API")

model, class_names, device = load_trained_model(CHECKPOINT_PATH, CLASSES_PATH)
transform = get_transform()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)

    top_prob, top_idx = probs.max(1)
    predicted_class = class_names[top_idx.item()]

    return {
        "class": predicted_class,
        "probability": float(top_prob.item()),
    }