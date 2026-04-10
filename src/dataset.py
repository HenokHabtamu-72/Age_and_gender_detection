from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from simple_transforms import SimpleImageTransform


class UTKFaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int = 128, train: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.train = train
        self.transform = self._build_transform()

    def _build_transform(self):
        return SimpleImageTransform(image_size=self.image_size, train=self.train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        gender = torch.tensor(float(row["gender"]), dtype=torch.float32)
        age_bucket = torch.tensor(int(row.get("age_bucket_idx", row.get("age_bucket_index", row.get("age_bucket_idx", 0)))), dtype=torch.long)

        if "age_bucket_idx" not in row.index:
            # Safe fallback for older CSVs.
            from utils import age_to_bucket_index
            age_bucket = torch.tensor(age_to_bucket_index(float(row["age"])), dtype=torch.long)

        sample = {
            "image": image,
            "age": age,
            "gender": gender,
            "age_bucket": age_bucket,
            "image_path": str(image_path),
            "filename": row["filename"],
        }
        return sample
