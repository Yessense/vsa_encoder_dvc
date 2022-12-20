import operator
from functools import reduce
from typing import Tuple

import torch
from torch import nn

def product(arr):
    return reduce(operator.mul, arr)

class Encoder(nn.Module):
    def __init__(self,
                 image_size: Tuple[int, int, int] = (3, 128, 128),
                 latent_dim: int = 1024,
                 n_features: int = 5,
                 hidden_channels: int = 64):
        super(Encoder, self).__init__()
        # Dataset parameters
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.image_size = image_size

        # NN parameters
        self.hidden_channels = hidden_channels
        self.in_channels = self.image_size[0]
        self.out_channels = 2 * self.latent_dim * self.n_features
        self.activation = torch.nn.ReLU()
        cnn_kwargs = dict(kernel_size=4, stride=2, padding=1)

        if image_size == (3, 128, 128):
            # Convolutional layers
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
            )
            self.reshape = (self.hidden_channels, 4, 4)
        elif image_size == (1, 64, 64):
            # Convolutional layers
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
                nn.Conv2d(self.hidden_channels, self.hidden_channels, **cnn_kwargs), nn.ReLU(),
            )
            self.reshape = (self.hidden_channels, 4, 4)
        else:
            raise ValueError("Wrong image size")

        self.final_layers = nn.Sequential(
            nn.Linear(product(self.reshape), self.out_channels), nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # x -> (batch_size, in_channels, width, height)
        x = self.cnn_layers(x)
        x = x.reshape((batch_size, -1))
        # x -> (batch_size, self.reshape)

        x = self.final_layers(x)
        x = x.reshape(-1, 2, self.latent_dim * self.n_features)

        mu, log_var = x.unbind(1)
        return mu, log_var