from __future__ import annotations

import torch

from config import Config


def build_standard_optimizer(model: torch.nn.Module, config: Config):
    params = model.parameters()
    if config.optimizer == "gd":
        return torch.optim.SGD(params, lr=config.lr, momentum=0.0, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    return None
