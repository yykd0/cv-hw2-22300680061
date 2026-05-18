from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(csv_paths: list[Path], metric: str, out: Path) -> None:
    plt.figure(figsize=(8, 4.5), dpi=160)
    for path in csv_paths:
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        label = path.parent.name
        x = df["epoch"] if "epoch" in df.columns else range(len(df))
        plt.plot(x, df[metric], label=label, linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.grid(alpha=0.25)
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plot_metric(args.csv, args.metric, args.out)


if __name__ == "__main__":
    main()

