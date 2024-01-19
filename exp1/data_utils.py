import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import random_split


def get_dataloaders(batch_size):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR10 dataset split.
    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform,
    )
    dataset_train, dataset_val = random_split(dataset_train, [45000, 5000])
    dataset_test = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform,
    )
    # Create data loaders.
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset_val, 
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, val_loader, test_loader