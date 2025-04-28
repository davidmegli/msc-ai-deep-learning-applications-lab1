'''
Author: David Megli
Date: 2025-04-28
File: utils.py
Description: Utility functions for dataset loading.
'''
import torch
from torchvision import datasets, transforms

def get_data_loaders(config):
    dataset_name = config['dataset']['name'].lower()
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset'].get('num_workers', 2)

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader