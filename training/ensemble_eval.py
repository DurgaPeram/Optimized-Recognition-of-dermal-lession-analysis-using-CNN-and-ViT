import torch
import timm
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import HAM10000Dataset, get_val_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- BASIC SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIRS = [
    "../data/raw/HAM10000_images_part_1",
    "../data/raw/HAM10000_images_part_2"
]

# ---------------- SWIN VAL TRANSFORM ----------------
def swin_val_tfms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

# ---------------- DATASET ----------------
effnet_ds = HAM10000Dataset("val.csv", IMAGE_DIRS, get_val_transforms())
swin_ds   = HAM10000Dataset("val.csv", IMAGE_DIRS, swin_val_tfms())

effnet_loader = DataLoader(effnet_ds, batch_size=16, shuffle=False, num_workers=0)
swin_loader   = DataLoader(swin_ds, batch_size=16, shuffle=False, num_workers=0)

# ---------------- LOAD MODELS ----------------
# EfficientNet
effnet = models.efficientnet_v2_s(weights=None)
effnet.classifier[1] = torch.nn.Linear(effnet.classifier[1].in_features, 7)
effnet.load_state_dict(torch.load("../models/effnet.pt", map_location=device))
effnet.to(device)
effnet.eval()

# Swin
swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=7
)
swin.load_state_dict(torch.load("../models/swin.pt", map_location=device))
swin.to(device)
swin.eval()

# ---------------- ENSEMBLE EVALUATION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for (x1, y), (x2, _) in zip(effnet_loader, swin_loader):
        x1, x2 = x1.to(device), x2.to(device)

        logits_eff = effnet(x1)
        logits_swin = swin(x2)

        logits = (logits_eff + logits_swin) / 2.0
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

# ---------------- METRICS ----------------
acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average="macro")

print("\nENSEMBLE RESULTS")
print("-----------------------------")
print(f"Accuracy : {acc:.4f}")
print(f"Macro F1 : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
