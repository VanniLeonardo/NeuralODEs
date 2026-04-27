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


def main() -> None:
    """Entry point: loads config, initializes wandb, runs the training loop."""

    # 1. Initialize Configuration and parse overrides
    config = ODEConfig()
    config.hidden_dim = 2  # Force hidden_dim to 2 for vizualization of failure modes
    config.epochs = 100 
    parser = argparse.ArgumentParser(description="Neural ODE training")
    for field, value in config.__dict__.items():
        parser.add_argument(f"--{field}", type=type(value), default=value)
    args = parser.parse_args()
    for field in config.__dict__:
        setattr(config, field, getattr(args, field))

    # 2. Initialize Weights & Biases
    wandb.init(project="neural-odes-30562", config=config.__dict__)

    # 3. Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Running on device: [bold]{device}[/bold]")

    # 4. Load Data
    train_loader, val_loader = get_dataloaders(
        dataset=config.dataset,
        n_samples=config.n_samples,
        batch_size=config.batch_size,
        val_split=config.val_split,
        noise=config.noise,
    )

    # 5. Initialize Model
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

    # 6. Training Loop
    for epoch in range(config.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        wandb.log({"epoch": epoch, **train_metrics})

        if config.hidden_dim == 2 and epoch % 10 == 0:
            visualize_2d_features(model, train_loader, device, epoch)

        if epoch % 20 == 0:
            plot_ode_flows(model, train_loader, device, epoch)

        if epoch % 5 == 0:
            console.log(
                f"Epoch {epoch:>3} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['accuracy']:.3f} | "
                f"Fwd NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
                f"Bwd NFE: {train_metrics.get('backward_nfe_mean', 0):.1f}"
            )

    wandb.finish()


if __name__ == "__main__":
    main()
