from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_pet_classification_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    image_size = int(data_cfg.get("image_size", 224))
    download = bool(data_cfg.get("download", True))
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(cfg["train"]["batch_size"])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    trainval = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=download,
    )
    test = datasets.OxfordIIITPet(
        root=root,
        split="test",
        target_types="category",
        transform=eval_tf,
        download=download,
    )
    val_len = max(1, int(len(trainval) * float(data_cfg.get("val_ratio", 0.15))))
    train_len = len(trainval) - val_len
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_set, val_set = random_split(trainval, [train_len, val_len], generator=generator)
    val_set.dataset = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        transform=eval_tf,
        download=False,
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


class PetSegTransform:
    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        mask = mask.convert("L")
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.train:
            scale = random.uniform(0.85, 1.15)
            size = int(self.image_size * scale)
            image = TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [size, size], interpolation=InterpolationMode.NEAREST)
            image = TF.center_crop(image, [self.image_size, self.image_size])
            mask = TF.center_crop(mask, [self.image_size, self.image_size])
        else:
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        image_t = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_arr = np.asarray(mask, dtype=np.int64)
        mask_t = torch.from_numpy(np.clip(mask_arr - 1, 0, 2)).long()
        return image_t, mask_t


class OxfordPetSegmentation(Dataset):
    def __init__(self, root: str | Path, split: str, transform: PetSegTransform, download: bool = True) -> None:
        self.base = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.base[index]
        return self.transform(image, mask)


def build_pet_segmentation_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    image_size = int(data_cfg.get("image_size", 256))
    download = bool(data_cfg.get("download", True))
    num_workers = int(data_cfg.get("num_workers", 4))
    batch_size = int(cfg["train"]["batch_size"])
    train_tf = PetSegTransform(image_size, train=True)
    eval_tf = PetSegTransform(image_size, train=False)
    trainval = OxfordPetSegmentation(root, "trainval", train_tf, download=download)
    val_len = max(1, int(len(trainval) * float(data_cfg.get("val_ratio", 0.15))))
    train_len = len(trainval) - val_len
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_set, val_set = random_split(trainval, [train_len, val_len], generator=generator)
    val_set.dataset = OxfordPetSegmentation(root, "trainval", eval_tf, download=False)
    test_set = OxfordPetSegmentation(root, "test", eval_tf, download=download)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )

