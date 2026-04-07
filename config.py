from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json


CLASSIFICATION_DATASETS = {"mnist", "cifar10"}
REGRESSION_DATASETS = {"california_housing"}
CNN_ONLY_DATASETS = {"mnist", "cifar10"}
VALID_MODELS = {"linear", "mlp1", "mlp3", "cnn"}
VALID_OPTIMIZERS = {"gd", "sgd", "adam", "svrg", "mb_svrg"}


@dataclass
class Config:
    dataset: str = "mnist"
    model: str = "mlp1"
    optimizer: str = "sgd"
    seed: int = 42
    device: str = "cuda"
    data_root: str = "./data"
    output_root: str = "./results"

    batch_size: int = 128
    eval_batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.9

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    svrg_snapshot_freq: int = 1
    svrg_inner_steps: Optional[int] = None
    svrg_batch_size: int = 128

    hidden_dim: int = 100
    mlp3_dims: tuple[int, int, int] = (256, 128, 64)
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    cnn_fc_dim: int = 256
    dropout: float = 0.0

    val_ratio: float = 0.1
    num_workers: int = 2
    pin_memory: bool = True

    log_interval: int = 50
    save_curves: bool = True
    save_checkpoints: bool = False

    def validate(self) -> None:
        self.dataset = self.dataset.lower()
        self.model = self.model.lower()
        self.optimizer = self.optimizer.lower()

        if self.dataset not in CLASSIFICATION_DATASETS | REGRESSION_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.model not in VALID_MODELS:
            raise ValueError(f"Unsupported model: {self.model}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        if self.model == "cnn" and self.dataset not in CNN_ONLY_DATASETS:
            raise ValueError("CNN is only valid for MNIST and CIFAR-10.")
        if self.dataset == "california_housing" and self.model == "linear":
            # For regression this behaves as linear regression, which is valid.
            pass
        if self.optimizer == "gd" and self.batch_size != 0:
            # batch_size is ignored by GD but accepted.
            pass
        if self.optimizer in {"svrg", "mb_svrg"} and self.svrg_batch_size <= 0:
            raise ValueError("svrg_batch_size must be positive for SVRG variants.")
        if self.optimizer in {"sgd", "adam", "mb_svrg"} and self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

    @property
    def task_type(self) -> str:
        return "classification" if self.dataset in CLASSIFICATION_DATASETS else "regression"

    @property
    def run_name(self) -> str:
        return f"{self.dataset}_{self.model}_{self.optimizer}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["task_type"] = self.task_type
        data["run_name"] = self.run_name
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def apply_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config field: {key}")
        setattr(config, key, value)
    config.validate()
    return config
