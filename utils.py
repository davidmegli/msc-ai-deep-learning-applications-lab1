'''
Author: David Megli
Date: 2025-04-28
File: utils.py
Description: Utility functions for loading configurations, models, optimizers, and loss functions.
'''
import yaml
import torch
from torch import optim
import torch.nn as nn
from torchvision import models

from models.model import *

def load_config(config_path: str):
    """ Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
            dict: Configuration dictionary.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(model_config):
    """ Get a model instance based on the configuration.
    Args:
        model_config (dict): Configuration dictionary for the model.
    Returns:
        nn.Module: An instance of the model.
    Raises:
        ValueError: If the model name is unknown.
    """

    model_name = model_config['name'].lower()
    model_params = model_config.get('params', {})
    # Model zoo from which to select the model
    model_zoo = {
        'simplemlp': SimpleMLP,
        'parametrizedmlp': ParametrizedMLP,
        'residualmlp': ResidualMLP,
        'customcnn': CustomCNN,
        'simplecnn': SimpleCNN,
        'residualcnn': ResidualCNN,
    }
    if model_name not in model_zoo:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_zoo[model_name](**model_params)

    
def get_optimizer(optimizer_config, parameters):
    """ Get an optimizer instance based on the configuration.
    Args:
        optimizer_config (dict): Configuration dictionary for the optimizer.
        parameters (iterable): Model parameters to optimize.
    Returns:
        optim.Optimizer: An instance of the optimizer.
    """
    optimizer_name = optimizer_config['name'].lower()
    optimizer_params = optimizer_config.get('params', {})

    # Conversione esplicita dei parametri numerici da stringa a float
    if 'weight_decay' in optimizer_params:
        optimizer_params['weight_decay'] = float(optimizer_params['weight_decay'])

    if optimizer_name == 'adam':
        return optim.Adam(parameters, **optimizer_params)
    elif optimizer_name == 'sgd':
        return optim.SGD(parameters, **optimizer_params)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(parameters, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_loss(loss_config):
    """ Get a loss function instance based on the configuration.
    Args:
        loss_config (dict): Configuration dictionary for the loss function.
    Returns:
        nn.Module: An instance of the loss function.
    Raises:
        ValueError: If the loss function name is unknown.
    """
    loss_name = loss_config['name'].lower()
    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
def get_scheduler(scheduler_config, optimizer):
    """ Get a learning rate scheduler instance based on the configuration.
    Args:
        scheduler_config (dict): Configuration dictionary for the scheduler.
        optimizer (optim.Optimizer): The optimizer to schedule.
    Returns:
        optim.lr_scheduler: An instance of the learning rate scheduler.
    Raises:
        ValueError: If the scheduler name is unknown.
    """
    if scheduler_config is None:
        return None

    scheduler_name = scheduler_config['name'].lower()
    scheduler_params = scheduler_config.get('params', {})

    if scheduler_name == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_name == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")