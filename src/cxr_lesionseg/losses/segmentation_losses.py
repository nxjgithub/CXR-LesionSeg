# src/cxr_lesionseg/losses/segmentation_losses.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Input:
        logits: [B, 1, H, W]
        targets: [B, 1, H, W]
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        return loss.mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss + DiceLoss.

    pos_weight is used to increase the loss weight of foreground pixels.
    It is useful for highly imbalanced binary lesion segmentation.
    """

    def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            smooth: float = 1e-6,
            pos_weight: float | None = None,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            pos_weight = logits.new_tensor([float(self.pos_weight)])
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight,
            )

        dice = self.dice_loss(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice