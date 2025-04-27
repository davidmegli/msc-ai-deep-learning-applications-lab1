'''
Author: David Megli
Date: 2025-04-28
'''
import yaml
import torch
from torch import optim
import torch.nn as nn
from torchvision import models

from models.model import SimpleMLP

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(name: str, num_classes: int, model_params: dict = {}):
    if name.lower() == "simple_mlp":
        return SimpleMLP(
            input_dim=model_params.get("input_dim", 784),
            hidden_dim=model_params.get("hidden_dim", 256),
            num_classes=num_classes
        )

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

def get_loss(loss_name: str):
    if loss_name.lower() == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_name.lower() == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss '{loss_name}' not supported.")
    
def get_scheduler(scheduler_name: str, optimizer, scheduler_params: dict):
    if scheduler_name.lower() == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get('mode', 'min'),
            factor=scheduler_params.get('factor', 0.1),
            patience=scheduler_params.get('patience', 10),
            verbose=True
        )
    elif scheduler_name.lower() == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', 10),
            eta_min=scheduler_params.get('eta_min', 0)
        )
    elif scheduler_name.lower() == "exponentiallr":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_params.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not supported.")