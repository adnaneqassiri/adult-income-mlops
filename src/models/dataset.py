import torch
from torch.utils.data import Dataset


class AdultDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(
            X.values,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y.values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]