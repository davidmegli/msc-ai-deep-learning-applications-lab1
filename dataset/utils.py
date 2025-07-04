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
import numpy as np

def get_data_loaders(config):
    """ Creates DataLoaders for training, validation, and test sets based on the configuration.
    Args:
        config (dict): Configuration dictionary containing dataset parameters.
    Returns:
        tuple: A tuple containing DataLoaders for training, validation, and test sets.
    Raises:
        ValueError: If the dataset name is unknown.
    """
    
    dataset_name = config['dataset']['name'].lower()
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset'].get('num_workers', 2)
    test_size = config['dataset'].get('test_size', 0.1)  # percentuale test
    val_size = config['dataset'].get('val_size', 0.1)    # percentuale validation
    device = config['trainer'].get('device', None)
    pin_memory = True if device == 'cuda' else False

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
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        full_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Split manuale
    #targets = full_dataset.targets.numpy() # When loading CIFAR-10: "AttributeError: 'list' object has no attribute 'numpy'""
    if isinstance(full_dataset.targets, torch.Tensor): # Per gestire sia Tensor che list
        targets = full_dataset.targets.numpy()
    else:
        targets = np.array(full_dataset.targets)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
