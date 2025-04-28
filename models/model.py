'''
Author: David Megli
Date: 2025-04-28
'''
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))  # Flatten 👍
    

class ParametrizedMLP(nn.Module):
    def __init__(self, layer_sizes):
        super(ParametrizedMLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) -1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))