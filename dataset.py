import pandas as pd
import torch
from torch.utils.data import Dataset

from labels import LETTER_TO_INDEX


class LandmarkDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.labels = df["label"].map(LETTER_TO_INDEX).values
        self.features = df.drop(columns=["label"]).values.astype("float32")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label