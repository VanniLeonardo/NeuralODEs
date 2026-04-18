import torch
import torch.nn as nn
import wandb
from models.networks import ODENet
from training.engine import train_epoch
from config import ODEConfig
from data.synthetic import get_concentric_circles

def main():
    # 1. Initialize Configuration
    config = ODEConfig()

    config.hidden_dim = 2  # FORCE the ODE to stay in 2D to match the concentric circles dataset as in Chen et al. section 5.3
    
    # 2. Initialize Weights & Biases
    wandb.init(project="neural-odes-30562", config=config.__dict__)
    
    # 3. Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 4. Initialize Model
    model = ODENet(
        data_dim=config.in_features,
        hidden_dim=config.hidden_dim,
        num_classes=2,
        solver_type=config.solver_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    # 5. Load synthetic dataset
    dataloader = get_concentric_circles(
        batch_size=config.batch_size,
        n_samples=1000
    )

    # 6. Training Loop
    for epoch in range(config.epochs):
        train_metrics = train_epoch(model, dataloader, optimizer, criterion, device)
        
        # Log to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "forward_nfe": train_metrics["nfe"]
        })
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {train_metrics['loss']:.4f} | NFE: {train_metrics['nfe']:.1f}")

    wandb.finish()

if __name__ == "__main__":
    main()