'''
Author: David Megli
Date: 2025-04-28
File: train.py
Description: Main training script
'''
import argparse
import yaml
import os
import random
import torch
import numpy as np
import wandb

from utils import get_model, get_loss, get_optimizer, get_scheduler
from dataset.utils import get_data_loaders
from trainer import Trainer
from datetime import datetime
import time

def train(config_file_path, disable_wandb=False):

    # Load config
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Save copy of config
    os.makedirs(config['output_dir'], exist_ok=True)
    config_save_path = os.path.join(config['output_dir'], f"used_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.safe_dump(config, f)

    # Initialize dataset
    train_loader, val_loader = get_data_loaders(config)

    # Initialize model
    model = get_model(config['model'])

    # Loss, optimizer, scheduler
    criterion = get_loss(config['loss'])
    optimizer = get_optimizer(config['optimizer'], model.parameters())
    scheduler = get_scheduler(config.get('scheduler'), optimizer)
    
    # Device setup
    device = torch.device(config['trainer'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    # Weights & Biases
    use_wandb = config['trainer'].get('use_wandb', False) if not disable_wandb else False
    if use_wandb:
        wandb.init(project=config['project_name'], config=config, dir='./runs')

    start_time = time.time()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        output_dir=config['output_dir'],
        max_epochs=config['trainer']['epochs'],
        patience=config['trainer'].get('patience', 10),
        mixed_precision=config['trainer'].get('mixed_precision', False),
        project_name=config['project_name'],
        use_wandb=use_wandb,
        run_name=config['trainer'].get('run_name', 'default_run'),
        resume=config['trainer'].get('resume', False)
    )

    # Start training
    train_losses, train_accuracies, val_losses, val_accuracies = trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time

    metrics = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'run_name': config['trainer'].get('run_name', 'default_run'),
        'parameters': trainer.total_params,
        'training_time': training_time
    }

    metrics_path = os.path.join(config['output_dir'], "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved at {metrics_path}")

    return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_mlp.yaml', help='Path to config file')
    args = parser.parse_args()
    config_file_path = args.config
    train(config_file_path)