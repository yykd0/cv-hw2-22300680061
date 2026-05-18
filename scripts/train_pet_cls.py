from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn
from tqdm import tqdm

from cvhw2.data.pets import build_pet_classification_loaders
from cvhw2.metrics import accuracy
from cvhw2.models.attention import attach_attention_to_resnet
from cvhw2.utils import AverageMeter, MetricLogger, ensure_dir, load_config, seed_everything


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    arch = model_cfg["arch"]
    num_classes = int(model_cfg["num_classes"])
    pretrained = bool(model_cfg.get("pretrained", True))
    attention = model_cfg.get("attention", "none")
    if arch in {"resnet18", "resnet34"}:
        from torchvision import models

        weights = "IMAGENET1K_V1" if pretrained else None
        model = getattr(models, arch)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = attach_attention_to_resnet(model, attention)
        return model
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Install timm to run ViT/Swin experiments: pip install timm") from exc
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    lr_backbone = float(train_cfg.get("lr_backbone", train_cfg.get("lr", 3e-4)))
    lr_head = float(train_cfg.get("lr_head", lr_backbone))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    head_names = ["fc", "head", "classifier"]
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(h) for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, target in pbar:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits.detach(), target), batch_size)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")
    return {"train_loss": loss_meter.avg, "train_acc": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, name: str) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for images, target in tqdm(loader, desc=name, leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, target), batch_size)
    return {f"{name}_loss": loss_meter.avg, f"{name}_acc": acc_meter.avg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["output_dir"])
    train_loader, val_loader, test_loader = build_pet_classification_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["train"]["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda") if cfg["train"].get("amp", True) and device.type == "cuda" else None
    logger = MetricLogger(
        output_dir,
        backend=cfg.get("logging", {}).get("backend", "csv"),
        project=cfg.get("logging", {}).get("project"),
        run_name=cfg.get("logging", {}).get("run_name"),
        config=cfg,
    )
    best_acc = -1.0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        metrics = {"epoch": epoch}
        metrics.update(train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch))
        metrics.update(evaluate(model, val_loader, criterion, device, "val"))
        scheduler.step()
        logger.log(metrics, step=epoch)
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / "last.pt")
        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / "best.pt")
        print(metrics)
    test_metrics = evaluate(model, test_loader, criterion, device, "test")
    logger.log({"epoch": "test", **test_metrics})
    logger.close()
    print(f"Best val acc: {best_acc:.4f}; test: {test_metrics}")


if __name__ == "__main__":
    main()

