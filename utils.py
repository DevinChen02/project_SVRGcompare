from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_history_csv(history: Dict[str, List[float]], path: str | Path) -> None:
    import pandas as pd

    pd.DataFrame(history).to_csv(path, index=False)


def plot_curves(history: Dict[str, List[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    curve_groups = [
        ("loss", [k for k in ["train_loss", "val_loss", "test_loss"] if k in history]),
        ("metric", [k for k in ["train_metric", "val_metric", "test_metric"] if k in history]),
    ]

    for suffix, keys in curve_groups:
        if not keys:
            continue
        plt.figure(figsize=(7, 5))
        for key in keys:
            plt.plot(history[key], label=key)
        plt.xlabel("Epoch")
        plt.ylabel(suffix.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))
        plt.close()
