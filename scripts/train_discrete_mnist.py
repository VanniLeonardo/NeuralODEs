import argparse
import torch
import torch.nn as nn
import wandb

from data.dataloaders import get_mnist_dataloaders
from models.networks import DiscreteResNet
from training.engine import train_epoch


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

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return {"loss": avg_loss, "accuracy": accuracy}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--use_wandb_sweep",
        action="store_true",
        help="Enable WandB sweep mode (online, config pulled from wandb.config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Defaults from CLI / manual runs
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    lr = args.lr
    epochs = args.epochs

    default_config = {
        "model": "DiscreteResNet",
        "dataset": "MNIST",
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "lr": lr,
        "epochs": epochs,
    }

    # WandB init:
    # - offline for normal/manual runs
    # - online for sweep runs
    if args.use_wandb_sweep:
        wandb.init(
            project="neural-odes-30562",
            config=default_config,
        )

        # Override defaults with sweep values
        batch_size = wandb.config.batch_size
        hidden_dim = wandb.config.hidden_dim
        num_layers = wandb.config.num_layers
        lr = wandb.config.lr
        epochs = wandb.config.epochs

    else:
        wandb.init(
            mode="offline",
            project="neural-odes-30562",
            config=default_config,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    model = DiscreteResNet(
        data_dim=784,
        hidden_dim=hidden_dim,
        num_classes=10,
        num_layers=num_layers,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        best_test_acc = max(best_test_acc, test_metrics["accuracy"])

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "best_test_accuracy": best_test_acc,
                "num_parameters": n_params,
                "memory_mb": train_metrics["memory_mb"],
            }
        )

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch} | "
                f"train_loss: {train_metrics['loss']:.4f} | "
                f"train_acc: {train_metrics['accuracy']:.4f} | "
                f"test_loss: {test_metrics['loss']:.4f} | "
                f"test_acc: {test_metrics['accuracy']:.4f} | "
                f"best_test_acc: {best_test_acc:.4f} | "
                f"memory_mb: {train_metrics['memory_mb']:.1f}"
            )

    wandb.finish()


if __name__ == "__main__":
    main()
