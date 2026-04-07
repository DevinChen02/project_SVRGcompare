from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from config import Config


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_eval_loader: DataLoader
    train_dataset_tensor: TensorDataset
    input_shape: Tuple[int, ...]
    output_dim: int
    num_classes: int | None


def _make_loader(dataset, batch_size: int, shuffle: bool, config: Config) -> DataLoader:
    if batch_size <= 0:
        batch_size = len(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _build_california_housing(config: Config) -> DatasetBundle:
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.seed
    )
    val_size = config.val_ratio / (1.0 - 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=config.seed
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)
    X_test = x_scaler.transform(X_test).astype(np.float32)

    y_train = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val = y_scaler.transform(y_val).astype(np.float32)
    y_test = y_scaler.transform(y_test).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    return DatasetBundle(
        train_loader=_make_loader(train_ds, config.batch_size, True, config),
        val_loader=_make_loader(val_ds, config.eval_batch_size, False, config),
        test_loader=_make_loader(test_ds, config.eval_batch_size, False, config),
        train_eval_loader=_make_loader(train_ds, config.eval_batch_size, False, config),
        train_dataset_tensor=train_ds,
        input_shape=(X_train.shape[1],),
        output_dim=1,
        num_classes=None,
    )


def _mnist_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _cifar10_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def _split_train_val(dataset, config: Config):
    total_len = len(dataset)
    val_len = int(total_len * config.val_ratio)
    train_len = total_len - val_len
    generator = torch.Generator().manual_seed(config.seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def _stack_dataset_to_tensor(dataset) -> TensorDataset:
    xs, ys = [], []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    y = torch.as_tensor(ys)
    return TensorDataset(X, y)


def _build_mnist(config: Config) -> DatasetBundle:
    train_full = datasets.MNIST(root=config.data_root, train=True, download=True, transform=_mnist_transforms())
    test_ds = datasets.MNIST(root=config.data_root, train=False, download=True, transform=_mnist_transforms())
    train_ds, val_ds = _split_train_val(train_full, config)
    train_tensor_ds = _stack_dataset_to_tensor(train_ds)
    return DatasetBundle(
        train_loader=_make_loader(train_ds, config.batch_size, True, config),
        val_loader=_make_loader(val_ds, config.eval_batch_size, False, config),
        test_loader=_make_loader(test_ds, config.eval_batch_size, False, config),
        train_eval_loader=_make_loader(train_tensor_ds, config.eval_batch_size, False, config),
        train_dataset_tensor=train_tensor_ds,
        input_shape=(1, 28, 28),
        output_dim=10,
        num_classes=10,
    )


def _build_cifar10(config: Config) -> DatasetBundle:
    train_full = datasets.CIFAR10(root=config.data_root, train=True, download=True, transform=_cifar10_transforms())
    test_ds = datasets.CIFAR10(root=config.data_root, train=False, download=True, transform=_cifar10_transforms())
    train_ds, val_ds = _split_train_val(train_full, config)
    train_tensor_ds = _stack_dataset_to_tensor(train_ds)
    return DatasetBundle(
        train_loader=_make_loader(train_ds, config.batch_size, True, config),
        val_loader=_make_loader(val_ds, config.eval_batch_size, False, config),
        test_loader=_make_loader(test_ds, config.eval_batch_size, False, config),
        train_eval_loader=_make_loader(train_tensor_ds, config.eval_batch_size, False, config),
        train_dataset_tensor=train_tensor_ds,
        input_shape=(3, 32, 32),
        output_dim=10,
        num_classes=10,
    )


def build_datasets(config: Config) -> DatasetBundle:
    builders: Dict[str, callable] = {
        "california_housing": _build_california_housing,
        "mnist": _build_mnist,
        "cifar10": _build_cifar10,
    }
    return builders[config.dataset](config)
