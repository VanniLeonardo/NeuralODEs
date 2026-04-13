import torch
import torch.nn as nn
import wandb
from typing import Dict


def train_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> Dict[str, float]:
    """Trains the model for a single epoch."""
    model.train()
    total_loss = 0.0
    total_nfe = 0
    correct = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total_samples += x.size(0)
        
        # Track NFE if the model is continuous
        if hasattr(model, 'ode_func'):
            total_nfe += model.ode_func.nfe

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    avg_nfe = total_nfe / len(dataloader)  # Average NFE per batch

    return {"loss": avg_loss, "accuracy": accuracy, "nfe": avg_nfe}