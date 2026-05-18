from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        maxv = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * torch.sigmoid(avg + maxv)
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_map, max_map], dim=1))


class AttentionWrapper(nn.Module):
    def __init__(self, block: nn.Module, attention: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.block(x))


def _block_channels(block: nn.Module) -> int:
    if hasattr(block, "conv3"):
        return block.conv3.out_channels
    if hasattr(block, "conv2"):
        return block.conv2.out_channels
    raise ValueError(f"Cannot infer output channels for block {block.__class__.__name__}")


def attach_attention_to_resnet(model: nn.Module, kind: str) -> nn.Module:
    kind = kind.lower()
    if kind in {"none", "", "null"}:
        return model
    if kind not in {"se", "cbam"}:
        raise ValueError(f"Unsupported attention kind: {kind}")
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for idx, block in enumerate(layer):
            channels = _block_channels(block)
            attention = SEBlock(channels) if kind == "se" else CBAMBlock(channels)
            layer[idx] = AttentionWrapper(block, attention)
    return model

