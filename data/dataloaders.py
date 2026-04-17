from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def flatten_tensor(x):
    return x.view(-1)


def get_mnist_dataloaders(batch_size: int, data_root: str = "./data"):
    """
    Returns train and test dataloaders for MNIST.

    IMPORTANT:
    Images are flattened from [1, 28, 28] to [784]
    because the model uses linear layers (MLP).
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(flatten_tensor)
    ])

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader