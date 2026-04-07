from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import torch

from config import Config, apply_overrides
from datasets import build_datasets
from models import build_model
from trainer import Trainer
from utils import set_seed


DEFAULT_GRID = {
    "dataset": ["california_housing", "mnist", "cifar10"],
    "model": ["linear", "mlp1", "mlp3", "cnn"],
    "optimizer": ["gd", "sgd", "adam", "svrg", "mb_svrg"],
}


def valid_combo(dataset: str, model: str) -> bool:
    if dataset == "california_housing" and model == "cnn":
        return False
    return True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for dataset, model, optimizer in itertools.product(
        DEFAULT_GRID["dataset"], DEFAULT_GRID["model"], DEFAULT_GRID["optimizer"]
    ):
        if not valid_combo(dataset, model):
            continue
        cfg = Config(dataset=dataset, model=model, optimizer=optimizer, device=str(device))
        if dataset == "cifar10":
            cfg.epochs = 15
        if optimizer == "gd":
            cfg.batch_size = 0
        cfg.validate()

        print(f"\n=== Running {cfg.run_name} ===")
        set_seed(cfg.seed)
        bundle = build_datasets(cfg)
        model_obj = build_model(cfg, bundle.input_shape, bundle.output_dim)
        trainer = Trainer(
            model=model_obj,
            config=cfg,
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            test_loader=bundle.test_loader,
            train_eval_loader=bundle.train_eval_loader,
            train_dataset_tensor=bundle.train_dataset_tensor,
            device=device,
        )
        summary = trainer.fit()
        rows.append(summary)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "all_runs_summary.csv", index=False)
    with open(out_dir / "all_runs_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved aggregate results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
