from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from config import Config, apply_overrides
from datasets import build_datasets
from models import build_model
from trainer import Trainer
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Optimization comparison project: GD/SGD/Adam/SVRG")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--dataset", type=str, default=None, choices=["california_housing", "mnist", "cifar10"])
    parser.add_argument("--model", type=str, default=None, choices=["linear", "mlp1", "mlp3", "cnn"])
    parser.add_argument("--optimizer", type=str, default=None, choices=["gd", "sgd", "adam", "svrg", "mb_svrg"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--svrg-inner-steps", type=int, default=None)
    parser.add_argument("--svrg-batch-size", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cli_overrides(args) -> Dict[str, Any]:
    mapping = {
        "dataset": args.dataset,
        "model": args.model,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "data_root": args.data_root,
        "output_root": args.output_root,
        "device": args.device,
        "svrg_inner_steps": args.svrg_inner_steps,
        "svrg_batch_size": args.svrg_batch_size,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main():
    args = parse_args()
    cfg = Config()
    yaml_cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, yaml_cfg)
    cfg = apply_overrides(cfg, cli_overrides(args))

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    bundle = build_datasets(cfg)
    model = build_model(cfg, bundle.input_shape, bundle.output_dim)
    trainer = Trainer(
        model=model,
        config=cfg,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        train_eval_loader=bundle.train_eval_loader,
        train_dataset_tensor=bundle.train_dataset_tensor,
        device=device,
    )
    summary = trainer.fit()
    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {Path(cfg.output_root) / cfg.run_name}")


if __name__ == "__main__":
    main()
