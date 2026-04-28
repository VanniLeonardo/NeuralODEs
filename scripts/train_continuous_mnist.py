import torch
import torch.nn as nn
import wandb
import argparse

from data.dataloaders import get_mnist_dataloaders
from models.networks import ODENet, ConvODENet
from training.engine import train_epoch
from scripts.plot_fig3 import evaluate_tolerances, plot_figure_3


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return {"loss": total_loss / total_samples, "accuracy": correct / total_samples}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--solver", type=str, default="dopri5")
    parser.add_argument(
        "--network_type", type=str, default="cnn", choices=["mlp", "cnn"]
    )
    return parser.parse_args()


def main():
    args = parse_args()

    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    lr = args.lr
    epochs = args.epochs
    solver_type = args.solver
    network_type = args.network_type
    # Select architecture based on network_type:
    # - "mlp": fair comparison with DiscreteResNet (same input representation)
    # - "cnn": image-based architecture, closer to the original Neural ODE paper setup

    wandb.init(
        mode="offline",
        project="neural-odes-30562",
        config={
            "model": "ODENet",
            "network_type": network_type,
            "dataset": "MNIST",
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "epochs": epochs,
            "solver": solver_type,
        },
    )
    # Uncomment the following lines if running a WandB Sweep.
    # In that case, hyperparameters are automatically provided by WandB instead of CLI arguments.
    batch_size = wandb.config.batch_size
    hidden_dim = wandb.config.hidden_dim
    lr = wandb.config.lr
    epochs = wandb.config.epochs
    solver_type = wandb.config.solver
    network_type = wandb.config.network_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} | Network: {network_type.upper()}")

    # Determine if we should flatten the image based on network type
    flatten_img = network_type == "mlp"
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size, flatten=flatten_img
    )

    # Initialize the correct architecture
    if network_type == "mlp":
        model = ODENet(
            data_dim=784, hidden_dim=hidden_dim, num_classes=10, solver_type=solver_type
        ).to(device)
    else:
        model = ConvODENet(
            in_channels=1,
            num_filters=hidden_dim,
            num_classes=10,
            solver_type=solver_type,
        ).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "num_parameters": n_params,
                "forward_nfe": train_metrics.get("forward_nfe_mean", 0),
                "peak_memory_mb": train_metrics.get("memory_mb", 0.0),
            }
        )

        print(
            f"Epoch {epoch} | "
            f"train_acc: {train_metrics['accuracy']:.4f} | "
            f"test_acc: {test_metrics['accuracy']:.4f} | "
            f"NFE: {train_metrics.get('forward_nfe_mean', 0):.1f} | "
            f"Mem: {train_metrics.get('memory_mb', 0.0):.1f} MB"
        )

    # ==========================================================
    # POST-TRAINING: GENERATE FIGURE 3 TOLERANCE PLOTS
    # ==========================================================
    print("Evaluating solver tolerances to generate Figure 3...")
    # Grab a single batch from the test loader
    x_val, y_val = next(iter(test_loader))
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Run the rigorous mathematical evaluation
    results = evaluate_tolerances(model, x_val, y_val)

    # Plot and upload to WandB!
    plot_figure_3(results, epoch=epochs)
    print("Figure 3 generated and uploaded to WandB!")

    wandb.finish()


if __name__ == "__main__":
    main()
