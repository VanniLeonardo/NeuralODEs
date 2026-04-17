import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles


def get_1d_crossing_data(batch_size: int):
    x = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def get_concentric_circles(batch_size: int, n_samples: int = 1024, noise: float = 0.05):
    x, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader