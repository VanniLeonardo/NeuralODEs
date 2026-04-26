import numpy as np
import torch
from sklearn.datasets import make_circles as sklearn_make_circles
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


def get_1d_crossing_data(batch_size: int) -> DataLoader:
    """Returns a 1D crossing dataset for sanity-checking trajectory crossings."""
    x = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


# get_concentric_circles uses sklearn, returns a DataLoader — used by main.py.
# make_circles uses numpy with a fixed seed, returns raw tensors -- used by
# get_dataloaders for reproducible train/val splits across the solver ablation.
def get_concentric_circles(
    batch_size: int,
    n_samples: int = 1024,
    noise: float = 0.05,
) -> DataLoader:
    """Returns a DataLoader of concentric circles via sklearn.

    Args:
        batch_size (int): Batch size for the loader.
        n_samples (int): Total number of points.
        noise (float): Std of Gaussian noise added to coordinates.
    """
    x, y = sklearn_make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def make_circles(
    n_samples: int = 1000,
    noise: float = 0.05,
    factor: float = 0.5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generates two concentric circles for binary classification.

    Points are sampled uniformly on $[0, 2\pi)$; the inner circle is scaled
    by `factor`. Standard toy benchmark from Chen et al. (2018).

    Args:
        n_samples (int): Total number of points across both classes.
        noise (float): Std of Gaussian noise added to $(x, y)$ coordinates.
        factor (float): Inner circle radius as fraction of outer ($r_{\text{inner}}$).
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Sample angles uniformly on $[0, 2\pi)$
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)

    outer = np.stack([np.cos(theta_outer), np.sin(theta_outer)], axis=1)
    inner = np.stack([np.cos(theta_inner), np.sin(theta_inner)], axis=1) * factor

    X = np.concatenate([outer, inner], axis=0) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_spirals(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generates two interleaving Archimedean spirals for binary classification.

    Radial coordinate $r = \theta / (4\pi)$ with $\theta \in [0, 4\pi]$.
    Class 1 is rotated by $\pi$ relative to class 0.

    Args:
        n_samples (int): Total number of points across both classes.
        noise (float): Std of Gaussian noise added to $(x, y)$ coordinates.
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2

    # $r = \theta / (4\pi)$, so $r \in [0, 1]$ as $\theta$ spans $[0, 4\pi]$
    theta = np.linspace(0, 4 * np.pi, n_per_class)
    r = theta / (4 * np.pi)

    x0 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x1 = np.stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)], axis=1)

    X = np.concatenate([x0, x1], axis=0) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Generates two interleaving crescent moons for binary classification.

    Upper moon spans $\theta \in [0, \pi]$; lower moon is offset by $(1, -0.5)$.

    Args:
        n_samples (int): Total number of points across both classes.
        noise (float): Std of Gaussian noise added to $(x, y)$ coordinates.
        seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    # Upper moon: $\theta \in [0, \pi]$
    theta_upper = np.linspace(0, np.pi, n_upper)
    upper = np.stack([np.cos(theta_upper), np.sin(theta_upper)], axis=1)

    # Lower moon offset by $(1, -0.5)$
    theta_lower = np.linspace(0, np.pi, n_lower)
    lower = np.stack([1 - np.cos(theta_lower), -np.sin(theta_lower) - 0.5], axis=1)

    X = np.concatenate([upper, lower], axis=0) + rng.normal(0, noise, (n_samples, 2))
    y = np.concatenate([np.zeros(n_upper), np.ones(n_lower)]).astype(np.int64)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
