import torch
import timm
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models

# ---------- APP ----------
app = FastAPI(title="DermaVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LABELS ----------
CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanocytic Nevus",
    "Vascular Lesion",
    "Melanoma"
]

# ---------- LOAD MODELS ----------
effnet = models.efficientnet_v2_s(weights=None)
effnet.classifier[1] = torch.nn.Linear(effnet.classifier[1].in_features, 7)
effnet.load_state_dict(torch.load("../models/effnet.pt", map_location=device))
effnet.to(device).eval()

swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=7
)
swin.load_state_dict(torch.load("../models/swin.pt", map_location=device))
swin.to(device).eval()

# ---------- TRANSFORMS ----------
effnet_tfms = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

swin_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ---------- API ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_np = np.array(image)

    x1 = effnet_tfms(image=image_np)["image"].unsqueeze(0).to(device)
    x2 = swin_tfms(image=image_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = (effnet(x1) + swin(x2)) / 2
        probs = torch.softmax(logits, dim=1)[0]

    idx = probs.argmax().item()

    return {
        "prediction": CLASS_NAMES[idx],
        "confidence": float(probs[idx]) * 100
    }
