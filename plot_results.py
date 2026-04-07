from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True, help="CSV summary file.")
    parser.add_argument("--out-dir", type=str, default="plots")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset]
        metric_col = "best_test_metric"
        plt.figure(figsize=(8, 5))
        for model_name in sorted(sub["model"].unique()):
            model_df = sub[sub["model"] == model_name].sort_values("optimizer")
            plt.plot(model_df["optimizer"], model_df[metric_col], marker="o", label=model_name)
        plt.xlabel("Optimizer")
        plt.ylabel(metric_col)
        plt.title(f"{dataset}: optimizer comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}_optimizer_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()
