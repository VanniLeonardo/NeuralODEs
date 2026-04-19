import torch
import torch.nn as nn
import wandb

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

def main():
    # --- DEFAULTS (Overridden by wandb sweep if running) ---
    batch_size = 64
    hidden_dim = 64          # Keep this around 64 for Conv so params don't explode
    lr = 1e-3
    epochs = 30
    solver_type = "dopri5"
    network_type = "mlp"     # Toggle this to "mlp" or "cnn"

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
            "solver": solver_type
        },
    )
    
    # Pull from config in case a Sweep is managing the run
    batch_size = wandb.config.batch_size
    hidden_dim = wandb.config.hidden_dim
    lr = wandb.config.lr
    epochs = wandb.config.epochs
    solver_type = wandb.config.solver
    network_type = wandb.config.network_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} | Network: {network_type.upper()}")

    # Determine if we should flatten the image based on network type
    flatten_img = (network_type == "mlp")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size, flatten=flatten_img)

    # Initialize the correct architecture
    if network_type == "mlp":
        model = ODENet(data_dim=784, hidden_dim=hidden_dim, num_classes=10, solver_type=solver_type).to(device)
    else:
        model = ConvODENet(in_channels=1, num_filters=hidden_dim, num_classes=10, solver_type=solver_type).to(device)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "num_parameters": n_params,
            "forward_nfe": train_metrics["nfe"],
            "peak_memory_mb": train_metrics["memory_mb"]
        })

        print(
            f"Epoch {epoch} | "
            f"train_acc: {train_metrics['accuracy']:.4f} | "
            f"test_acc: {test_metrics['accuracy']:.4f} | "
            f"NFE: {train_metrics['nfe']:.1f} | "
            f"Mem: {train_metrics['memory_mb']:.1f} MB"
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