import argparse
import torch
import torch.nn as nn
import wandb
from rich.console import Console

from config import ODEConfig
from data.dataloaders import get_dataloaders
from models.networks import ODENet
from training.engine import train_epoch
from training.utils import plot_ode_flows, visualize_2d_features

console = Console()


def get_gpu_memory_stats(device: torch.device) -> dict:
    """Return GPU memory usage in MB. Empty dict when not on CUDA."""
    if device.type != "cuda":
        return {}

    gpu_id = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated(gpu_id) / (1024**2),
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved(gpu_id) / (1024**2),
        "gpu_mem_peak_allocated_mb": torch.cuda.max_memory_allocated(gpu_id)
        / (1024**2),
    }


def main() -> None:
    """Entry point: loads config, initializes wandb, runs the training loop."""

    config = ODEConfig()
    config.hidden_dim = 2  # Force hidden_dim to 2 for vizualization of failure modes
    config.epochs = 200
    parser = argparse.ArgumentParser(description="Neural ODE training")
    for field, value in config.__dict__.items():
        parser.add_argument(f"--{field}", type=type(value), default=value)
    args = parser.parse_args()
    for field in config.__dict__:
        setattr(config, field, getattr(args, field))

    wandb.init(project="neural-odes-30562", config=config.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Running on device: [bold]{device}[/bold]")

    train_loader, val_loader = get_dataloaders(
        dataset=config.dataset,
        n_samples=config.n_samples,
        batch_size=config.batch_size,
        val_split=config.val_split,
        noise=config.noise,
    )

    model = ODENet(
        data_dim=config.in_features,
        hidden_dim=config.hidden_dim,
        num_classes=2,
        solver_type=config.solver_type,
        atol=config.atol,
        rtol=config.rtol,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        if device.type == "cuda":
            gpu_id = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            torch.cuda.reset_peak_memory_stats(gpu_id)

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        gpu_mem = get_gpu_memory_stats(device)

        wandb.log({"epoch": epoch, **train_metrics, **gpu_mem})

        if config.hidden_dim == 2 and epoch % 10 == 0:
            visualize_2d_features(model, train_loader, device, epoch)

        if epoch % 20 == 0:
            plot_ode_flows(model, train_loader, device, epoch)

        if epoch % 5 == 0:
            msg = (
                f"Epoch {epoch:>3} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.3f} | "
                f"Fwd NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
                f"Bwd NFE: {train_metrics.get('backward_nfe_mean', 0):.1f}"
            )

            if gpu_mem:
                msg += (
                    f" | GPU Alloc: {gpu_mem['gpu_mem_allocated_mb']:.1f} MB"
                    f" | GPU Reserved: {gpu_mem['gpu_mem_reserved_mb']:.1f} MB"
                    f" | GPU Peak: {gpu_mem['gpu_mem_peak_allocated_mb']:.1f} MB"
                )

            console.log(msg)

    wandb.finish()


if __name__ == "__main__":
    main()
