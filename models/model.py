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
    def __init__(self, layer_sizes, activation='ReLU'):
        super(ParametrizedMLP, self).__init__()
        layers = []

        # Mapping attivazioni
        activation_layer = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'ELU': nn.ELU,
            'Sigmoid': nn.Sigmoid,
            'Tanh': nn.Tanh
        }

        act_fn = activation_layer.get(activation, nn.ReLU)

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_fn())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class ResidualBlock(nn.Module):
    def __init__(self, layer_sizes, activation='ReLU'):
        super(ResidualBlock, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # No activation after last linear
                layers.append(getattr(nn, activation)())
        self.block = nn.Sequential(*layers)

        # Per la skip connection dobbiamo assicurarci che input e output abbiano stessa dimensione
        self.need_projection = layer_sizes[0] != layer_sizes[-1]
        if self.need_projection:
            self.projection = nn.Linear(layer_sizes[0], layer_sizes[-1])

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.need_projection:
            identity = self.projection(identity)
        return out + identity

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, output_dim, activation='ReLU'):
        super(ResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualBlock([hidden_dim, hidden_dim], activation) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x