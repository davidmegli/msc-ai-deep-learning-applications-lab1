'''
Author: David Megli
Date: 2025-04-28
'''
import yaml
import torch
from torch import optim
from torchvision import models

from models.model import SimpleMLP  # Import dei modelli custom

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(name: str, num_classes: int):
    # Prima cerchiamo nei modelli custom
    if name.lower() == "simple_mlp":
        return SimpleMLP(input_dim=784, hidden_dim=256, num_classes=num_classes)
    
    # Poi nei modelli torchvision
    elif name.lower() == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Model '{name}' not supported.")

def get_optimizer(optimizer_name: str, model_params, lr: float):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")
