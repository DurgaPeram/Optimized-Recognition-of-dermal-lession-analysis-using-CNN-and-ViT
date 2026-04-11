import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch.cuda.amp import autocast, GradScaler

from dataset import HAM10000Dataset, get_train_transforms, get_val_transforms
from losses import FocalLoss

# ---------------- BASIC SETUP ----------------
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIRS = [
    "../data/raw/HAM10000_images_part_1",
    "../data/raw/HAM10000_images_part_2"
]

# ---------------- DATASETS ----------------
train_ds = HAM10000Dataset(
    "train.csv",
    IMAGE_DIRS,
    transform=get_train_transforms()
)

val_ds = HAM10000Dataset(
    "val.csv",
    IMAGE_DIRS,
    transform=get_val_transforms()
)

train_loader = DataLoader(
    train_ds,
    batch_size=12,          # STABLE for RTX 3050 laptop
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=12,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ---------------- MODEL ----------------
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 7
)
model.to(device)

# ---------------- FREEZE BACKBONE (STAGE 1) ----------------
for param in model.features.parameters():
    param.requires_grad = False

# ---------------- LOSS + OPTIMIZER ----------------
class_weights = torch.tensor(
    [0.9431, 0.6000, 0.2806, 2.6816, 0.0460, 2.1717, 0.2771],
    dtype=torch.float
).to(device)

criterion = FocalLoss(alpha=class_weights, gamma=2.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=15
)

scaler = GradScaler()

# ---------------- TRAINING CONFIG ----------------
FREEZE_EPOCHS = 5
FINETUNE_EPOCHS = 15
best_acc = 0.0

# ---------------- STAGE 1: TRAIN CLASSIFIER ----------------
for epoch in range(FREEZE_EPOCHS):
    model.train()
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"[FREEZE] Epoch {epoch+1}/{FREEZE_EPOCHS} completed")
    torch.cuda.empty_cache()

# ---------------- UNFREEZE BACKBONE ----------------
for param in model.features.parameters():
    param.requires_grad = True

# ---------------- STAGE 2: FULL FINETUNING ----------------
for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # -------- VALIDATION --------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    scheduler.step()

    print(f"[FINETUNE] Epoch {epoch+1}/{FINETUNE_EPOCHS} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../models/effnet.pt")

    torch.cuda.empty_cache()

print(f"\nBest EfficientNet Validation Accuracy: {best_acc:.4f}")
