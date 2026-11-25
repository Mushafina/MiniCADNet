"""MiniCADNet: Miniature Compact Automotive Damage Classification Network.

Custom lightweight CNN built from scratch using standard PyTorch layers
(no pretrained weights, transfer learning, or external model architectures).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 conv + pointwise 1x1 conv block with optional 2x2 max pooling."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MiniCADNet(nn.Module):
    """Compact 5-class automotive damage classifier for 96x96 RGB crops."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block1 = DepthwiseSeparableBlock(16, 24, pool=True)
        self.block2 = DepthwiseSeparableBlock(24, 32, pool=True)
        self.block3 = DepthwiseSeparableBlock(32, 48, pool=False)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)
