import torch
from data.dataloaders import get_mnist_dataloaders
from models.networks import DiscreteResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_mnist_dataloaders(batch_size=64)

model = DiscreteResNet(
    data_dim=784,
    hidden_dim=128,
    num_classes=10,
    num_layers=5
).to(device)

x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

logits = model(x)

print("input shape:", x.shape)
print("labels shape:", y.shape)
print("output shape:", logits.shape)
print("output dtype:", logits.dtype)