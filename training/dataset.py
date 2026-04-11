import os
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dirs, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dirs = image_dirs
        self.transform = transform

        self.class_map = {
            'akiec': 0,
            'bcc': 1,
            'bkl': 2,
            'df': 3,
            'nv': 4,
            'vasc': 5,
            'mel': 6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        label = self.class_map[row['dx']]

        img_path = None
        for d in self.image_dirs:
            p = os.path.join(d, img_id + ".jpg")
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            raise FileNotFoundError(f"{img_id}.jpg not found")

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ---------- TRANSFORMS (OUTSIDE CLASS) ----------

def get_train_transforms():
    return A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(384, 384),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
