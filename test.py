'''
Author: David Megli
Date: 2025-04-28
'''
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import SimpleMLP
from utils import load_config, get_model, get_loss

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy

def main(config_path):
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if config['dataset_name'].lower() == 'mnist':
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {config['dataset_name']} not supported yet in test.py.")

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = get_model(
        name=config['model_name'],
        num_classes=config['num_classes'],
        model_params=config.get('model_params', {})
    )
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pth')))
    model.to(device)

    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Optional: log su WandB
    if config.get('use_wandb', False):
        import wandb
        wandb.init(project=config['project_name'], name=f"{config['run_name']}_test")
        wandb.log({"Test Accuracy": accuracy})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_test.yaml', help='Path to config file')
    args = parser.parse_args()

    main(args.config)
