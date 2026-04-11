import torch
import timm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import HAM10000Dataset

# ---------------- BASIC SETUP ----------------
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIRS = [
    "../data/raw/HAM10000_images_part_1",
    "../data/raw/HAM10000_images_part_2"
]

# ---------------- TRANSFORMS (224x224) ----------------
def train_tfms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

def val_tfms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

# ---------------- DATASETS ----------------
train_ds = HAM10000Dataset("train.csv", IMAGE_DIRS, train_tfms())
val_ds   = HAM10000Dataset("val.csv", IMAGE_DIRS, val_tfms())

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ---------------- MODEL ----------------
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=7
)
model.to(device)

# ---------------- FREEZE BACKBONE ----------------
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

# ---------------- LOSS (CROSS ENTROPY ONLY) ----------------
class_weights = torch.tensor(
    [0.9431, 0.6000, 0.2806, 2.6816, 0.0460, 2.1717, 0.2771],
    dtype=torch.float
).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

scaler = GradScaler()

# ---------------- STAGE 1: WARM-UP ----------------
FREEZE_EPOCHS = 5
for epoch in range(FREEZE_EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast():
            loss = criterion(model(x), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"[WARMUP] Epoch {epoch+1}/{FREEZE_EPOCHS} completed")
    torch.cuda.empty_cache()

# ---------------- UNFREEZE MODEL ----------------
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20
)

# ---------------- STAGE 2: FULL FINETUNE ----------------
best_acc = 0.0
FINETUNE_EPOCHS = 20

for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast():
            loss = criterion(model(x), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # -------- VALIDATION --------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    scheduler.step()

    print(f"[FINETUNE] Epoch {epoch+1}/{FINETUNE_EPOCHS} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "../models/swin.pt")

    torch.cuda.empty_cache()

print(f"\nBest Swin Validation Accuracy: {best_acc:.4f}")
