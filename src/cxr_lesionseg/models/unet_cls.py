# src/cxr_lesionseg/models/unet_cls.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from .unet import DoubleConv, Down, Up


class UNetWithClassificationHead(nn.Module):
    """
    U-Net with an image-level classification head.

    Outputs:
        {
            "seg_logits": [B, 1, H, W],
            "cls_logits": [B, 1]
        }
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            base_channels: int = 32,
            cls_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        bottleneck_channels = base_channels * 16

        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=cls_dropout),
            nn.Linear(bottleneck_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        cls_logits = self.cls_head(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        seg_logits = self.outc(x)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits,
        }