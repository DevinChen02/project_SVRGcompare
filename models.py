from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from config import Config


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class LinearModel(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int):
        super().__init__()
        in_dim = math.prod(input_shape)
        self.net = nn.Sequential(Flatten(), nn.Linear(in_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP1(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        in_dim = math.prod(input_shape)
        layers = [Flatten(), nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP3(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int,
        dims: Tuple[int, int, int],
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = math.prod(input_shape)
        d1, d2, d3 = dims
        layers = [
            Flatten(),
            nn.Linear(in_dim, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Linear(d2, d3),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(d3, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int,
        channels: Tuple[int, int, int] = (32, 64, 128),
        fc_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        c_in = input_shape[0]
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(c_in, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        flattened = c3 * 4 * 4
        classifier = [Flatten(), nn.Linear(flattened, fc_dim), nn.ReLU()]
        if dropout > 0:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(fc_dim, output_dim))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(config: Config, input_shape: Tuple[int, ...], output_dim: int) -> nn.Module:
    if config.model == "linear":
        return LinearModel(input_shape, output_dim)
    if config.model == "mlp1":
        return MLP1(input_shape, output_dim, hidden_dim=config.hidden_dim, dropout=config.dropout)
    if config.model == "mlp3":
        return MLP3(input_shape, output_dim, dims=config.mlp3_dims, dropout=config.dropout)
    if config.model == "cnn":
        return SmallCNN(input_shape, output_dim, channels=config.cnn_channels, fc_dim=config.cnn_fc_dim, dropout=config.dropout)
    raise ValueError(f"Unknown model: {config.model}")
