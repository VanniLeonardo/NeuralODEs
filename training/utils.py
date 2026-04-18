import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import os

def visualize_2d_features(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, epoch: int):
    """Passes 2D data through the ODE and plots the deformed feature space."""
    model.eval()
    all_features = []
    all_labels =[]

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            # Pass data only up to the ODE block (skip the final fc layer)
            h = model.downsampling(x)
            h_T = model.ode_block(h)
            
            all_features.append(h_T.cpu())
            all_labels.append(y.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.scatter(features[labels==0, 0], features[labels==0, 1], color='red', alpha=0.5, label='Class 0')
    plt.scatter(features[labels==1, 0], features[labels==1, 1], color='blue', alpha=0.5, label='Class 1')
    plt.title(f"ODE Feature Space at Epoch {epoch}")
    plt.legend()
    
    # Save locally
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/features_epoch_{epoch}.png"
    plt.savefig(plot_path)
    
    # Upload to WandB
    wandb.log({"feature_space": wandb.Image(plot_path)}, commit=False)
    
    plt.close()