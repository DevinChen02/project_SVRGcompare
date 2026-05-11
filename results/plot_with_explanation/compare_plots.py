"""
Generates exactly 3 plot types, one file per dataset:
  1. Training loss P(w)  vs. #grad/n  (convergence speed)
  2. Test error          vs. #grad/n  (generalization quality)
  3. Gradient variance   vs. #grad/n  (variance proxy)

X-axis: #grad/n (effective passes through the data).
Y-axis: log scale for loss and variance to handle large magnitude differences
        between optimizers

#grad/n cost per epoch:
  GD, SGD, Adam  -> 1.0  (one full pass per epoch)
  SVRG           -> 2.0  (full snapshot pass + inner loop pass)
  Mini-batch SVRG-> 2.0  (same structure)

Usage:
    python compare_plots.py                    # reads ./results, saves to ./plots
    python compare_plots.py --results ./results --out ./plots
    python compare_plots.py --dataset mnist    # only one dataset
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


OPTIMIZER_STYLES: Dict[str, dict] = {
    # GD uses thick black line + large markers so it stays visible even when
    # nearly flat at the top of the log axis (GD barely converges on deep nets).
    "gd":      {"color": "#000000", "linestyle": "--",  "linewidth": 2.0,
                "marker": "D", "markersize": 5, "markevery": 1, "zorder": 10,
                "label": "GD (full-batch)"},
    "sgd":     {"color": "#E84A3A", "linestyle": "-",   "linewidth": 2.0,
                "marker": "o", "markersize": 5, "markevery": 1, "zorder": 3,
                "label": "SGD"},
    "adam":    {"color": "#2CA02C", "linestyle": "-",   "linewidth": 2.0,
                "marker": "s", "markersize": 5, "markevery": 1, "zorder": 4,
                "label": "Adam"},
    "svrg":    {"color": "#1F77B4", "linestyle": "-.",  "linewidth": 2.0,
                "marker": "^", "markersize": 5, "markevery": 1, "zorder": 5,
                "label": "SVRG"},
    "mb_svrg": {"color": "#9467BD", "linestyle": ":",   "linewidth": 2.2,
                "marker": "v", "markersize": 5, "markevery": 1, "zorder": 6,
                "label": "Mini-batch SVRG"},
}

OPT_DRAW_ORDER = ["sgd", "adam", "svrg", "mb_svrg", "gd"]

# #grad/n cost per epoch
GRAD_COST: Dict[str, float] = {
    "gd":      1.0,
    "sgd":     1.0,
    "adam":    1.0,
    "svrg":    2.0,
    "mb_svrg": 2.0,
}

MODEL_DISPLAY = {
    "linear": "Linear",
    "mlp1":   "MLP-1 (100)",
    "mlp3":   "MLP-3 (256-128-64)",
    "cnn":    "Small CNN",
}

DATASET_DISPLAY = {
    "california_housing": "California Housing (Regression)",
    "mnist":              "MNIST (Classification)",
    "cifar10":            "CIFAR-10 (Classification)",
}

OPT_ORDER = ["gd", "sgd", "adam", "svrg", "mb_svrg"]

def _style(opt: str) -> dict:
    return OPTIMIZER_STYLES.get(
        opt,
        {"color": "gray", "linestyle": "-", "linewidth": 1.5,
         "marker": "x", "markersize": 4, "markevery": 1, "label": opt}
    )


def _legend_order(opts) -> List[str]:
    """Order for legend entries (GD first)."""
    return [o for o in OPT_ORDER if o in opts] + \
           [o for o in opts if o not in OPT_ORDER]


def _draw_order(opts) -> List[str]:
    """Draw order: GD is plotted last so it renders on top of all other lines."""
    return [o for o in OPT_DRAW_ORDER if o in opts] + \
           [o for o in opts if o not in OPT_DRAW_ORDER]


def _grad_n_axis(optimizer: str, n_epochs: int) -> np.ndarray:
    """Return cumulative #grad/n for each epoch."""
    cost = GRAD_COST.get(optimizer, 1.0)
    return np.arange(1, n_epochs + 1) * cost


def _variance_proxy(loss_series: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Rolling std-dev of training loss as a proxy for gradient variance.
    A noisier optimizer produces larger loss fluctuations.
    """
    var = np.full_like(loss_series, np.nan)
    for i in range(len(loss_series)):
        start = max(0, i - window + 1)
        var[i] = np.std(loss_series[start: i + 1])
    return var


def _test_error(odf: pd.DataFrame, is_cls: bool) -> np.ndarray:
    """1 - accuracy for classification; RMSE for regression."""
    if is_cls:
        return 1.0 - odf["test_metric"].values
    return odf["test_metric"].values


def _is_classification(dataset: str) -> bool:
    return dataset in {"mnist", "cifar10"}


def _apply_log_yaxis(ax, values_list: List[np.ndarray], label: str) -> None:
    """
    Apply log scale when the value range spans more than one order of magnitude,
    ensuring all plotted data (including flat GD lines) remains visible.
    """
    all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in values_list
                               if len(v) > 0])
    if len(all_vals) == 0:
        return

    vmin, vmax = all_vals.min(), all_vals.max()
    use_log = (vmax / max(vmin, 1e-9)) > 10  # log scale if range > 10x

    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel(f"{label} (log scale)", fontsize=10)
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    else:
        ax.set_ylabel(label, fontsize=10)


def _plot_line(ax, x, y, opt: str) -> None:
    """Plot a single optimizer line with its style."""
    s = _style(opt)
    # Filter out non-positive values for log scale compatibility
    mask = np.isfinite(y) & (y > 0)
    if not mask.any():
        return
    ax.plot(
        x[mask], y[mask],
        label=s["label"],
        color=s["color"],
        linestyle=s["linestyle"],
        linewidth=s["linewidth"],
        marker=s["marker"],
        markersize=s["markersize"],
        markevery=s.get("markevery", 1),
        markerfacecolor=s["color"],
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=s.get("zorder", 2),
    )


def _reorder_legend(ax) -> None:
    """Reorder legend entries to match OPT_ORDER (GD first)."""
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {l: h for h, l in zip(handles, labels)}
    desired_labels = [OPTIMIZER_STYLES[o]["label"] for o in OPT_ORDER
                      if OPTIMIZER_STYLES[o]["label"] in label_to_handle]
    # Include any labels not in OPT_ORDER at the end
    extra = [l for l in labels if l not in desired_labels]
    ordered_labels = desired_labels + extra
    ordered_handles = [label_to_handle[l] for l in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, fontsize=8, framealpha=0.8)


def _finalize_ax(ax, title: str, show_ylabel: bool,
                 ylabel: str, n_epochs_max: int) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("#grad / n", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.grid(True, which="major", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))


def load_all_histories(results_dir: Path) -> pd.DataFrame:
    """
    Scan results/ for run directories containing history.csv.
    Model and optimizer names are explicitly listed to avoid greedy regex
    mismatching mb_svrg (which contains an underscore).
    """
    pattern = re.compile(
        r"^(?P<dataset>.+)"
        r"_(?P<model>linear|mlp1|mlp3|cnn)"
        r"_(?P<optimizer>mb_svrg|svrg|adam|sgd|gd)"
        r"_seed(?P<seed>\d+)$"
    )
    records = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = pattern.match(run_dir.name)
        if not m:
            continue
        hist_path = run_dir / "history.csv"
        if not hist_path.exists():
            continue
        try:
            hist = pd.read_csv(hist_path)
        except Exception as e:
            print(f"  Warning: cannot read {hist_path}: {e}")
            continue

        opt = m.group("optimizer")
        hist["dataset"]   = m.group("dataset")
        hist["model"]     = m.group("model")
        hist["optimizer"] = opt
        hist["seed"]      = int(m.group("seed"))
        hist["epoch"]     = np.arange(1, len(hist) + 1)
        hist["grad_n"]    = _grad_n_axis(opt, len(hist))
        records.append(hist)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)



# Plot 1: Training loss vs #grad/n

def plot_train_loss(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    ds_df = df[df["dataset"] == dataset]
    models = sorted(ds_df["model"].unique())

    fig, axes = plt.subplots(
        1, len(models),
        figsize=(5.5 * len(models), 5),
        squeeze=False
    )
    fig.suptitle(
        f"{DATASET_DISPLAY.get(dataset, dataset)}\n"
        f"Training Loss  P(w)  vs.  #grad / n",
        fontsize=13, fontweight="bold",
    )

    for col, model in enumerate(models):
        ax = axes[0][col]
        mdf = ds_df[ds_df["model"] == model]
        opts = _draw_order(mdf["optimizer"].unique())

        all_y = []
        plot_data = []
        for opt in opts:
            odf = mdf[mdf["optimizer"] == opt].sort_values("grad_n")
            if odf.empty:
                continue
            y = odf["train_loss"].values
            x = odf["grad_n"].values
            all_y.append(y)
            plot_data.append((x, y, opt))

        # Decide log scale before drawing
        all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in all_y]
                                  ) if all_y else np.array([])
        use_log = (len(all_vals) > 0 and
                   all_vals.max() / max(all_vals.min(), 1e-9) > 10)
        if use_log:
            ax.set_yscale("log")

        for x, y, opt in plot_data:
            _plot_line(ax, x, y, opt)
        _reorder_legend(ax)

        ylabel = "Training Loss (log)" if use_log else "Training Loss"
        _finalize_ax(ax, MODEL_DISPLAY.get(model, model),
                     col == 0, ylabel, int(mdf["epoch"].max()))

    plt.tight_layout()
    path = out_dir / f"{dataset}_1_train_loss_vs_gradn.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# Plot 2: Test error vs #grad/n

def plot_test_error(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    ds_df = df[df["dataset"] == dataset]
    models = sorted(ds_df["model"].unique())
    is_cls = _is_classification(dataset)
    base_ylabel = "Test Error  (1 − Accuracy)" if is_cls else "Test RMSE"

    fig, axes = plt.subplots(
        1, len(models),
        figsize=(5.5 * len(models), 5),
        squeeze=False
    )
    fig.suptitle(
        f"{DATASET_DISPLAY.get(dataset, dataset)}\n"
        f"Test Error  vs.  #grad / n",
        fontsize=13, fontweight="bold",
    )

    for col, model in enumerate(models):
        ax = axes[0][col]
        mdf = ds_df[ds_df["model"] == model]
        opts = _draw_order(mdf["optimizer"].unique())

        all_y = []
        plot_data = []
        for opt in opts:
            odf = mdf[mdf["optimizer"] == opt].sort_values("grad_n")
            if odf.empty:
                continue
            y = _test_error(odf, is_cls)
            x = odf["grad_n"].values
            all_y.append(y)
            plot_data.append((x, y, opt))

        all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in all_y]
                                  ) if all_y else np.array([])
        use_log = (len(all_vals) > 0 and
                   all_vals.max() / max(all_vals.min(), 1e-9) > 10)
        if use_log:
            ax.set_yscale("log")

        for x, y, opt in plot_data:
            _plot_line(ax, x, y, opt)

        ylabel = f"{base_ylabel} (log)" if use_log else base_ylabel
        _finalize_ax(ax, MODEL_DISPLAY.get(model, model),
                     col == 0, ylabel, int(mdf["epoch"].max()))

    plt.tight_layout()
    path = out_dir / f"{dataset}_2_test_error_vs_gradn.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# Plot 3: Variance proxy vs #grad/n

def plot_variance(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    ds_df = df[df["dataset"] == dataset]
    models = sorted(ds_df["model"].unique())

    fig, axes = plt.subplots(
        1, len(models),
        figsize=(5.5 * len(models), 5),
        squeeze=False
    )
    fig.suptitle(
        f"{DATASET_DISPLAY.get(dataset, dataset)}\n"
        f"Gradient Variance (proxy)  vs.  #grad / n",
        fontsize=13, fontweight="bold",
    )

    for col, model in enumerate(models):
        ax = axes[0][col]
        mdf = ds_df[ds_df["model"] == model]
        opts = _draw_order(mdf["optimizer"].unique())

        all_y = []
        plot_data = []
        for opt in opts:
            odf = mdf[mdf["optimizer"] == opt].sort_values("grad_n")
            if odf.empty:
                continue
            y = _variance_proxy(odf["train_loss"].values)
            x = odf["grad_n"].values
            all_y.append(y)
            plot_data.append((x, y, opt))

        all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in all_y]
                                  ) if all_y else np.array([])
        use_log = (len(all_vals) > 0 and
                   all_vals.max() / max(all_vals.min(), 1e-9) > 10)
        if use_log:
            ax.set_yscale("log")

        for x, y, opt in plot_data:
            _plot_line(ax, x, y, opt)

        ylabel = "Loss Std Dev — variance proxy (log)" if use_log \
            else "Loss Std Dev — variance proxy"
        _finalize_ax(ax, MODEL_DISPLAY.get(model, model),
                     col == 0, ylabel, int(mdf["epoch"].max()))

    plt.tight_layout()
    path = out_dir / f"{dataset}_3_variance_vs_gradn.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

# Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--out",     type=str, default="./plots")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["california_housing", "mnist", "cifar10"])
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading histories from: {results_dir.resolve()}")
    df = load_all_histories(results_dir)

    if df.empty:
        print("No history.csv files found. Run run_experiments.py first.")
        return

    datasets = [args.dataset] if args.dataset else sorted(df["dataset"].unique())
    print(f"  Datasets  : {list(datasets)}")
    print(f"  Optimizers: {sorted(df['optimizer'].unique())}")
    print(f"  Models    : {sorted(df['model'].unique())}")
    print(f"Saving to   : {out_dir.resolve()}\n")

    for dataset in datasets:
        print(f"[{dataset}]")
        plot_train_loss(df, out_dir, dataset)
        plot_test_error(df, out_dir, dataset)
        plot_variance(df, out_dir, dataset)

    print(f"\nDone. {len(datasets) * 3} plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
