'''
Author: David Megli
Date: 2025-04-28
'''
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import load_config, get_model, get_optimizer, get_loss, get_scheduler
from trainer import Trainer
from datetime import datetime
import yaml

def main():
    # Argument parser per la configurazione
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, default="config_mlp.yaml", help="Path to config file.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Save config usato nella output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config['output_dir'], exist_ok=True)
    config_save_path = os.path.join(config['output_dir'], f"used_config_{timestamp}.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Setup dataset
    if config['model_name'].lower() == "simple_mlp":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif config['model_name'].lower() == "resnet18":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    else:
        raise ValueError(f"No dataset setup for model {config['model_name']}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Model
    model = get_model(config['model_name'], config['num_classes'], config.get('model_params', {}))

    # Loss
    criterion = get_loss(config['loss'])

    # Optimizer
    optimizer = get_optimizer(config['optimizer'], model.parameters(), config['learning_rate'])

    # Scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = get_scheduler(
            scheduler_name=config['scheduler_name'],
            optimizer=optimizer,
            scheduler_params=config.get('scheduler_params', {})
        )


    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        output_dir=config['output_dir'],
        max_epochs=config['max_epochs'],
        patience=config['patience'],
        mixed_precision=config['mixed_precision'],
        project_name=config['project_name'],
        use_wandb=config['use_wandb'],
        run_name=config['run_name'],
    )

    trainer.train()

if __name__ == "__main__":
    main()