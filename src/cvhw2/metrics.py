from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    mask = (target >= 0) & (target < num_classes)
    idx = num_classes * target[mask] + pred[mask]
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


@torch.no_grad()
def mean_iou_from_confmat(confmat: torch.Tensor) -> tuple[float, list[float]]:
    confmat = confmat.float()
    intersection = torch.diag(confmat)
    union = confmat.sum(dim=1) + confmat.sum(dim=0) - intersection
    iou = intersection / union.clamp_min(1.0)
    return iou.mean().item(), iou.tolist()

