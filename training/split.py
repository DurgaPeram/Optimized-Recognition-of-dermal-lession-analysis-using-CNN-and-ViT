import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

csv_path = "../data/raw/HAM10000_metadata.csv"
df = pd.read_csv(csv_path)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, val_idx in splitter.split(df, df["dx"]):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

print("Train:", len(train_df), "Val:", len(val_df))
