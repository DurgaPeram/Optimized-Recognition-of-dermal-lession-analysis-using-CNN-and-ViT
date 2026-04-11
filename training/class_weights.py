import pandas as pd
import numpy as np
import torch

csv_path = "../data/raw/HAM10000_metadata.csv"

df = pd.read_csv(csv_path)

class_map = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'nv': 4,
    'vasc': 5,
    'mel': 6
}

labels = df['dx'].map(class_map).values
class_counts = np.bincount(labels)

weights = 1.0 / class_counts
weights = weights / weights.sum() * len(class_counts)

class_weights = torch.tensor(weights, dtype=torch.float)

print("Class counts:", class_counts)
print("Class weights:", class_weights)
