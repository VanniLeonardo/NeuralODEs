import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3D projection
from typing import Any, Dict, List

_PLOTS_DIR: str = "plots"


def visualize_2d_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    """Passes 2D data through the ODE block and plots the deformed feature space.

    Args:
        model (nn.Module): Trained ODENet with hidden_dim == 2.
        dataloader (torch.utils.data.DataLoader): Source of input batches.
        device (torch.device): Device the model lives on.
        epoch (int): Current epoch, used for plot title and filename.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            h = model.downsampling(x)
            h_T = model.ode_block(h)
            all_features.append(h_T.cpu())
            all_labels.append(y.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        features[labels == 0, 0], features[labels == 0, 1],
        color="red", alpha=0.5, label="Class 0",
    )
    plt.scatter(
        features[labels == 1, 0], features[labels == 1, 1],
        color="blue", alpha=0.5, label="Class 1",
    )
    plt.title(f"ODE Feature Space at Epoch {epoch}")
    plt.legend()

    os.makedirs(_PLOTS_DIR, exist_ok=True)
    plot_path = f"{_PLOTS_DIR}/features_epoch_{epoch}.png"
    plt.savefig(plot_path)
    wandb.log({"feature_space": wandb.Image(plot_path)}, commit=False)
    plt.close()


def plot_ode_flows(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    is_anode: bool = False,
) -> None:
    """Plots the continuous ODE trajectories for a batch of data.

    Args:
        model (nn.Module): Trained ODENet; forward must support return_trajectory=True.
        dataloader (torch.utils.data.DataLoader): Source of one batch for plotting.
        device (torch.device): Device the model lives on.
        epoch (int): Current epoch, used for plot title and filename.
        is_anode (bool): If True, renders trajectories in 3D (augmented dim).
    """
    model.eval()
    x, y = next(iter(dataloader))
    x = x.to(device)

    with torch.no_grad():
        # trajectories shape: (time_steps, batch_size, hidden_dim)
        trajectories = model(x, return_trajectory=True).cpu().numpy()

    y = y.numpy()
    fig = plt.figure(figsize=(8, 6))

    if not is_anode:
        ax = fig.add_subplot(111)
        for i in range(len(x)):
            color = "blue" if y[i] == 1 else "red"
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1],
                color=color, alpha=0.3, linewidth=1,
            )
            ax.arrow(
                trajectories[-2, i, 0], trajectories[-2, i, 1],
                trajectories[-1, i, 0] - trajectories[-2, i, 0],
                trajectories[-1, i, 1] - trajectories[-2, i, 1],
                color=color, head_width=0.01, head_length=0.015, alpha=0.8,
            )
    else:
        ax = fig.add_subplot(111, projection="3d")
        for i in range(len(x)):
            color = "blue" if y[i] == 1 else "red"
            ax.plot(
                trajectories[:, i, 0], trajectories[:, i, 1], trajectories[:, i, 2],
                color=color, alpha=0.3,
            )
            ax.scatter(
                trajectories[-1, i, 0], trajectories[-1, i, 1], trajectories[-1, i, 2],
                color=color, s=10,
            )

    plt.title(f"ODE Flow Trajectories at Epoch {epoch}")
    os.makedirs(_PLOTS_DIR, exist_ok=True)
    plot_path = f"{_PLOTS_DIR}/flow_epoch_{epoch}.png"
    plt.savefig(plot_path)
    wandb.log({"ode_flow": wandb.Image(plot_path)}, commit=False)
    plt.close()


class NFEStats:
    r"""Accumulates per-batch forward and backward NFE across an epoch.

    The adjoint method solves a second ODE during backpropagation, so the
    total cost per batch is:
    $\text{NFE}_{\text{total}} = \text{NFE}_{\text{fwd}} + \text{NFE}_{\text{bwd}}$

    Tracking them separately reveals the relative cost of inference vs. training
    and how that ratio evolves as the vector field becomes smoother during training.
    """

    def __init__(self) -> None:
        self._forward: List[int] = []
        self._backward: List[int] = []

    def __len__(self) -> int:
        return len(self._forward)

    def record(self, forward: int, backward: int) -> None:
        """Appends NFE counts for a single batch.

        Args:
            forward (int): NFE during the forward ODE solve.
            backward (int): NFE during the adjoint ODE solve.
        """
        self._forward.append(forward)
        self._backward.append(backward)

    def summary(self) -> Dict[str, float]:
        """Mean and std of forward, backward, and total NFE across recorded batches."""
        fwd = np.array(self._forward, dtype=float)
        bwd = np.array(self._backward, dtype=float)
        tot = fwd + bwd
        return {
            "forward_nfe_mean": float(fwd.mean()),
            "forward_nfe_std": float(fwd.std()),
            "backward_nfe_mean": float(bwd.mean()),
            "backward_nfe_std": float(bwd.std()),
            "total_nfe_mean": float(tot.mean()),
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Any,
    path: str,
) -> None:
    """Saves model and optimizer state to disk.

    Args:
        model (nn.Module): The model to checkpoint.
        optimizer (torch.optim.Optimizer): The optimizer to checkpoint.
        epoch (int): Current epoch number, stored for resuming.
        config (Any): Config object serialized alongside the weights.
        path (str): Filepath to write the checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Loads model and optimizer state from a checkpoint file.

    Args:
        path (str): Filepath of the checkpoint to load.
        model (nn.Module): Model whose weights will be restored in-place.
        optimizer (torch.optim.Optimizer): Optimizer whose state will be restored.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])
