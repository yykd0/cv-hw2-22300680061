from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class MetricLogger:
    def __init__(self, output_dir: str | os.PathLike[str], backend: str = "csv", **kwargs: Any) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.backend = backend.lower()
        self.csv_path = self.output_dir / "metrics.csv"
        self._fieldnames: list[str] = []
        self._rows: list[dict[str, float | int | str]] = []
        self._client = None
        if self.backend == "wandb":
            import wandb

            self._client = wandb.init(
                project=kwargs.get("project", "cv-hw2"),
                name=kwargs.get("run_name"),
                config=kwargs.get("config"),
            )
        elif self.backend == "swanlab":
            import swanlab

            self._client = swanlab.init(
                project=kwargs.get("project", "cv-hw2"),
                experiment_name=kwargs.get("run_name"),
                config=kwargs.get("config"),
            )

    def log(self, metrics: dict[str, float | int | str], step: int | None = None) -> None:
        row = dict(metrics)
        if step is not None:
            row["step"] = step
        for key in row:
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(row)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for saved in self._rows:
                writer.writerow({k: saved.get(k, "") for k in self._fieldnames})
        if self.backend in {"wandb", "swanlab"} and self._client is not None:
            self._client.log(metrics, step=step)

    def close(self) -> None:
        if self.backend == "wandb" and self._client is not None:
            self._client.finish()
