'''
Author: David Megli
Date: 2025-04-28
File: utils.py
Description: Utility functions for dataset loading.
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def get_data_loaders(config):
    dataset_name = config['dataset']['name'].lower()
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset'].get('num_workers', 2)
    test_size = config['dataset'].get('test_size', 0.1)  # percentuale test
    val_size = config['dataset'].get('val_size', 0.1)    # percentuale validation

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split manuale
    targets = full_dataset.targets.numpy()
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)),
        test_size=(test_size + val_size),
        random_state=42,
        stratify=targets
    )

    temp_targets = targets[temp_idx]
    val_relative_size = val_size / (val_size + test_size)  # per il secondo split corretto

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_relative_size,
        random_state=42,
        stratify=temp_targets
    )

    # Sottodataset
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
