import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import os
from mpl_toolkits.mplot3d import Axes3D

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

def plot_ode_flows(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, epoch: int, is_anode: bool = False):
    """Plots the continuous ODE trajectories for a batch of data."""
    model.eval()
    x, y = next(iter(dataloader))
    x = x.to(device)
    
    with torch.no_grad():
        # trajectories shape: (time_steps, batch_size, hidden_dim)
        trajectories = model(x, return_trajectory=True).cpu().numpy()
        
    y = y.numpy()
    
    fig = plt.figure(figsize=(8, 6))
    
    # If standard NODE (2D), plot 2D flows
    if not is_anode:
        ax = fig.add_subplot(111)
        for i in range(len(x)):
            color = 'blue' if y[i] == 1 else 'red'
            # Plot the line across time for this specific sample
            ax.plot(trajectories[:, i, 0], trajectories[:, i, 1], color=color, alpha=0.3, linewidth=1)
            # Add an arrow at the end
            ax.arrow(trajectories[-2, i, 0], trajectories[-2, i, 1], 
                     trajectories[-1, i, 0] - trajectories[-2, i, 0], 
                     trajectories[-1, i, 1] - trajectories[-2, i, 1], 
                     color=color, head_width=0.05, alpha=0.8)
    
    # If ANODE (3D), plot 3D flows (per Beppe)
    else:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(x)):
            color = 'blue' if y[i] == 1 else 'red'
            ax.plot(trajectories[:, i, 0], trajectories[:, i, 1], trajectories[:, i, 2], color=color, alpha=0.3)
            ax.scatter(trajectories[-1, i, 0], trajectories[-1, i, 1], trajectories[-1, i, 2], color=color, s=10)
            
    plt.title(f"ODE Flow Trajectories at Epoch {epoch}")
    plot_path = f"plots/flow_epoch_{epoch}.png"
    plt.savefig(plot_path)
    wandb.log({"ode_flow": wandb.Image(plot_path)}, commit=False)
    plt.close()