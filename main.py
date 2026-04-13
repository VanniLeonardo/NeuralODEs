import torch
import torch.nn as nn
import wandb
from models.networks import ODENet
from training.engine import train_epoch
from config import ODEConfig

# NOTE: Member 1 will provide this import later
# from data.synthetic import get_dataloaders 

def main():
    # 1. Initialize Configuration
    config = ODEConfig()
    
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

    # --- MOCK DATA FOR NOW (Wait for Member 1) ---
    # We use random data just to ensure the pipeline doesn't crash
    x_dummy = torch.randn(1000, config.in_features)
    y_dummy = torch.randint(0, 2, (1000,))
    dataset = torch.utils.data.TensorDataset(x_dummy, y_dummy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # ---------------------------------------------

    # 5. Training Loop
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