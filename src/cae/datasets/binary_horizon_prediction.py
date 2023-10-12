import torch
from torch.utils.data import Dataset
import h5py
import math


class BinaryHorizonPredictionDataset(Dataset):
    def __init__(self, file_path, transform=None, subset_length=None, horizon=5):
        self.file_path = file_path
        self.transform = transform
        self.horizon = horizon
        with h5py.File(self.file_path, "r") as f:
            # Limit the dataset to only rows where there is a valid label
            total_length = len(f["images"])
            self.idxs = [
                start
                for start, end in zip(
                    range(0, total_length), range(horizon, total_length)
                )
                if not math.isnan(f["closes"][start])
                and not math.isnan(f["closes"][end])
            ]
            self.length = subset_length if subset_length is not None else len(self.idxs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        relative_idx = self.idxs[idx]
        with h5py.File(self.file_path, "r") as f:
            image = f["images"][relative_idx]
            label = (
                (0, 1)
                if f["closes"][relative_idx + self.horizon] > f["closes"][relative_idx]
                else (1, 0)
            )

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image), torch.tensor(label)
