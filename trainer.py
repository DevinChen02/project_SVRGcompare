from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import Config
from optimizers import build_standard_optimizer
from utils import AverageMeter, count_parameters, ensure_dir, plot_curves, save_history_csv, save_json


@dataclass
class EvalResult:
    loss: float
    metric: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        train_eval_loader: DataLoader,
        train_dataset_tensor: TensorDataset,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_eval_loader = train_eval_loader
        self.train_dataset_tensor = train_dataset_tensor
        self.device = device

        self.criterion = self._build_criterion()
        self.standard_optimizer = build_standard_optimizer(self.model, config)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_metric": [],
            "val_metric": [],
            "test_metric": [],
            "epoch_time_sec": [],
        }

        self.best_val_metric = float("-inf") if config.task_type == "classification" else float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.run_dir = ensure_dir(config.run_dir)

    def _build_criterion(self):
        if self.config.task_type == "classification":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()

    def _is_better(self, metric: float) -> bool:
        if self.config.task_type == "classification":
            return metric > self.best_val_metric
        return metric < self.best_val_metric

    def _compute_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        if self.config.task_type == "classification":
            preds = outputs.argmax(dim=1)
            return (preds == targets).float().mean().item()
        mse = torch.mean((outputs - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()

    def evaluate(self, loader: DataLoader) -> EvalResult:
        self.model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                if self.config.task_type == "regression":
                    outputs = outputs.view_as(y)
                loss = self.criterion(outputs, y)
                metric = self._compute_metric(outputs, y)
                n = x.size(0)
                loss_meter.update(loss.item(), n)
                metric_meter.update(metric, n)

        return EvalResult(loss=loss_meter.avg, metric=metric_meter.avg)

    def _full_batch_step(self) -> float:
        self.model.train()
        xs, ys = self.train_dataset_tensor.tensors
        x = xs.to(self.device)
        y = ys.to(self.device)
        self.standard_optimizer.zero_grad()
        outputs = self.model(x)
        if self.config.task_type == "regression":
            outputs = outputs.view_as(y)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.standard_optimizer.step()
        return loss.item()

    def _mini_batch_epoch(self) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        iterator = tqdm(self.train_loader, desc="train", leave=False)
        for step, (x, y) in enumerate(iterator, start=1):
            x = x.to(self.device)
            y = y.to(self.device)
            self.standard_optimizer.zero_grad()
            outputs = self.model(x)
            if self.config.task_type == "regression":
                outputs = outputs.view_as(y)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.standard_optimizer.step()
            loss_meter.update(loss.item(), x.size(0))
            if step % self.config.log_interval == 0:
                iterator.set_postfix(loss=f"{loss_meter.avg:.4f}")
        return loss_meter.avg

    def _batch_gradients(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        outputs = model(x)
        if self.config.task_type == "regression":
            outputs = outputs.view_as(y)
        loss = self.criterion(outputs, y)
        if self.config.weight_decay > 0:
            l2_penalty = 0.0
            for p in model.parameters():
                l2_penalty = l2_penalty + 0.5 * torch.sum(p ** 2)
            loss = loss + self.config.weight_decay * l2_penalty
        grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
        return [g.detach().clone() for g in grads]

    def _compute_full_gradient(self, snapshot_model: nn.Module) -> List[torch.Tensor]:
        snapshot_model.train()
        grad_sums = None
        total = 0
        for x, y in self.train_eval_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            grads = self._batch_gradients(snapshot_model, x, y)
            batch_n = x.size(0)
            if grad_sums is None:
                grad_sums = [g * batch_n for g in grads]
            else:
                for acc, g in zip(grad_sums, grads):
                    acc.add_(g * batch_n)
            total += batch_n
        assert grad_sums is not None
        return [g / total for g in grad_sums]

    @torch.no_grad()
    def _apply_grads(self, grads: List[torch.Tensor], lr: float):
        idx = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            p.add_(grads[idx], alpha=-lr)
            idx += 1

    def _svrg_epoch(self, mini_batch: bool) -> float:
        self.model.train()
        snapshot_model = copy.deepcopy(self.model).to(self.device)
        snapshot_model.eval()
        full_grad = self._compute_full_gradient(snapshot_model)

        if mini_batch:
            data_loader = DataLoader(
                self.train_dataset_tensor,
                batch_size=self.config.svrg_batch_size,
                shuffle=True,
                num_workers=0,
            )
        else:
            data_loader = DataLoader(
                self.train_dataset_tensor,
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )

        max_steps = self.config.svrg_inner_steps or len(data_loader)
        iterator = iter(data_loader)
        loss_meter = AverageMeter()

        for _ in tqdm(range(max_steps), desc="svrg", leave=False):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                x, y = next(iterator)
            x = x.to(self.device)
            y = y.to(self.device)

            curr_grads = self._batch_gradients(self.model, x, y)
            snap_grads = self._batch_gradients(snapshot_model, x, y)
            corrected = [gc - gs + fg for gc, gs, fg in zip(curr_grads, snap_grads, full_grad)]
            self._apply_grads(corrected, self.config.lr)

            with torch.no_grad():
                outputs = self.model(x)
                if self.config.task_type == "regression":
                    outputs = outputs.view_as(y)
                loss = self.criterion(outputs, y)
                loss_meter.update(loss.item(), x.size(0))

        return loss_meter.avg

    def fit(self) -> Dict:
        start_total = time.time()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            if self.config.optimizer == "gd":
                train_loss = self._full_batch_step()
            elif self.config.optimizer in {"sgd", "adam"}:
                train_loss = self._mini_batch_epoch()
            elif self.config.optimizer == "svrg":
                train_loss = self._svrg_epoch(mini_batch=False)
            elif self.config.optimizer == "mb_svrg":
                train_loss = self._svrg_epoch(mini_batch=True)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

            epoch_time = time.time() - epoch_start
            train_eval = self.evaluate(self.train_eval_loader)
            val_eval = self.evaluate(self.val_loader)
            test_eval = self.evaluate(self.test_loader)

            self.history["train_loss"].append(train_eval.loss if self.config.optimizer != "gd" else train_loss)
            self.history["val_loss"].append(val_eval.loss)
            self.history["test_loss"].append(test_eval.loss)
            self.history["train_metric"].append(train_eval.metric)
            self.history["val_metric"].append(val_eval.metric)
            self.history["test_metric"].append(test_eval.metric)
            self.history["epoch_time_sec"].append(epoch_time)

            metric_label = "acc" if self.config.task_type == "classification" else "rmse"
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={self.history['train_loss'][-1]:.4f} | "
                f"val_{metric_label}={val_eval.metric:.4f} | "
                f"test_{metric_label}={test_eval.metric:.4f} | "
                f"time={epoch_time:.2f}s"
            )

            if self._is_better(val_eval.metric):
                self.best_val_metric = val_eval.metric
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                if self.config.save_checkpoints:
                    torch.save(self.best_state, self.run_dir / "best_model.pt")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        best_test = self.evaluate(self.test_loader)
        total_time = time.time() - start_total

        summary = {
            "run_name": self.config.run_name,
            "dataset": self.config.dataset,
            "model": self.config.model,
            "optimizer": self.config.optimizer,
            "task_type": self.config.task_type,
            "num_parameters": count_parameters(self.model),
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "best_test_loss": best_test.loss,
            "best_test_metric": best_test.metric,
            "total_time_sec": total_time,
            "config": self.config.to_dict(),
        }

        save_json(summary, self.run_dir / "summary.json")
        save_history_csv(self.history, self.run_dir / "history.csv")
        if self.config.save_curves:
            plot_curves(self.history, self.run_dir / "curves.png")
        return summary
