import torch
from torch.utils.data import DataLoader
from torchvision import models
from dataset import HAM10000Dataset, get_train_transforms, get_val_transforms
from losses import FocalLoss
torch.backends.cudnn.benchmark = True


# paths
CSV_PATH = "../data/raw/HAM10000_metadata.csv"
IMAGE_DIRS = [
    "../data/raw/HAM10000_images_part_1",
    "../data/raw/HAM10000_images_part_2"
]

# dataset
train_ds = HAM10000Dataset(
    CSV_PATH, IMAGE_DIRS,
    transform=get_train_transforms()
)

val_ds = HAM10000Dataset(
    CSV_PATH, IMAGE_DIRS,
    transform=get_val_transforms()
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)


# model
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# class weights (paste from output)
class_weights = torch.tensor(
    [0.9431, 0.6000, 0.2806, 2.6816, 0.0460, 2.1717, 0.2771],
    dtype=torch.float
).to(device)

criterion = FocalLoss(alpha=class_weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# training loop (1 epoch test)
model.train()
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Batch {i}, Loss: {loss.item():.4f}")
    break


print("Training loop ran successfully")
