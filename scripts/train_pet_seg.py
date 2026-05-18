from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tqdm import tqdm

from cvhw2.data.pets import build_pet_segmentation_loaders
from cvhw2.losses import build_seg_loss
from cvhw2.metrics import confusion_matrix, mean_iou_from_confmat
from cvhw2.models.unet import UNet
from cvhw2.utils import AverageMeter, MetricLogger, ensure_dir, load_config, seed_everything


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch: int) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, mask in pbar:
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, mask)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")
    return {"train_loss": loss_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, name: str) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    confmat = torch.zeros(num_classes, num_classes, device=device)
    for images, mask in tqdm(loader, desc=name, leave=False):
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, mask)
        pred = logits.argmax(dim=1)
        confmat += confusion_matrix(pred, mask, num_classes).to(device)
        loss_meter.update(loss.item(), images.size(0))
    miou, class_iou = mean_iou_from_confmat(confmat.cpu())
    metrics = {f"{name}_loss": loss_meter.avg, f"{name}_miou": miou}
    for idx, value in enumerate(class_iou):
        metrics[f"{name}_iou_class_{idx}"] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg["output_dir"])
    train_loader, val_loader, test_loader = build_pet_segmentation_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=int(cfg["model"].get("in_channels", 3)),
        num_classes=int(cfg["model"].get("num_classes", 3)),
        base_channels=int(cfg["model"].get("base_channels", 32)),
    ).to(device)
    criterion = build_seg_loss(cfg["train"]["loss"], int(cfg["model"].get("num_classes", 3)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
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
    best_miou = -1.0
    num_classes = int(cfg["model"].get("num_classes", 3))
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        metrics = {"epoch": epoch}
        metrics.update(train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch))
        metrics.update(evaluate(model, val_loader, criterion, device, num_classes, "val"))
        scheduler.step()
        logger.log(metrics, step=epoch)
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / "last.pt")
        if metrics["val_miou"] > best_miou:
            best_miou = metrics["val_miou"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / "best.pt")
        print(metrics)
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes, "test")
    logger.log({"epoch": "test", **test_metrics})
    logger.close()
    print(f"Best val mIoU: {best_miou:.4f}; test: {test_metrics}")


if __name__ == "__main__":
    main()

