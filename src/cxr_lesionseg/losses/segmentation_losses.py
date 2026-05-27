# src/cxr_lesionseg/losses/segmentation_losses.py
# -*- coding: utf-8 -*-
"""
Next-stage binary segmentation losses for CXR-LesionSeg.

Drop-in replacement for:
    src/cxr_lesionseg/losses/segmentation_losses.py

Keeps the original DiceLoss and BCEDiceLoss, and adds:
    - TverskyLoss
    - FocalLoss
    - FocalTverskyLoss
    - DiceTverskyLoss
    - FocalDiceLoss
    - BCETverskyLoss
    - build_segmentation_loss(...)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Input:
        logits:  [B, 1, H, W]
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
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss + DiceLoss.

    pos_weight increases the foreground-pixel penalty. For SIIM-ACR, a moderate
    value such as 2.0 or 2.5 is usually safer than pushing it too high.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
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


class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced lesion segmentation.

    alpha controls FP penalty, beta controls FN penalty.
    For recall-oriented pneumothorax segmentation, beta > alpha is usually used,
    e.g. alpha=0.3, beta=0.7.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - score).mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss with logits.

    alpha is the positive-class weight. alpha=0.75 gives higher weight to
    foreground pixels, while gamma=2.0 down-weights easy pixels.
    """

    def __init__(
        self,
        alpha: Optional[float] = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)

        loss = (1.0 - pt).pow(self.gamma) * bce

        if self.alpha is not None:
            alpha_t = float(self.alpha) * targets + (1.0 - float(self.alpha)) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss.

    gamma < 1 increases the gradient around hard, low-overlap examples.
    Common setting for lesion segmentation: alpha=0.3, beta=0.7, gamma=0.75.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        score = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - score).pow(self.gamma).mean()


class DiceTverskyLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.tversky_weight * self.tversky(logits, targets)


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_alpha: Optional[float] = 0.75,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


class BCETverskyLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.3,
        tversky_weight: float = 0.7,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.pos_weight = pos_weight
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is None:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            pos_weight = logits.new_tensor([float(self.pos_weight)])
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        return self.bce_weight * bce + self.tversky_weight * self.tversky(logits, targets)


def build_segmentation_loss(
    loss_name: str = "bce_dice",
    bce_weight: float = 0.3,
    dice_weight: float = 0.7,
    tversky_weight: float = 0.5,
    focal_weight: float = 0.5,
    pos_weight: Optional[float] = 2.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_alpha: Optional[float] = 0.75,
    focal_gamma: float = 2.0,
    focal_tversky_gamma: float = 0.75,
    smooth: float = 1e-6,
) -> nn.Module:
    name = str(loss_name).strip().lower()

    if name in {"bce_dice", "bcedice", "bce+dice"}:
        return BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=smooth,
            pos_weight=pos_weight,
        )

    if name in {"dice_tversky", "dice+tversky", "tversky_dice"}:
        return DiceTverskyLoss(
            dice_weight=dice_weight,
            tversky_weight=tversky_weight,
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
        )

    if name in {"focal_dice", "focal+dice", "dice_focal"}:
        return FocalDiceLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            smooth=smooth,
        )

    if name in {"focal_tversky", "focaltversky"}:
        return FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_tversky_gamma,
            smooth=smooth,
        )

    if name in {"bce_tversky", "bce+tversky"}:
        return BCETverskyLoss(
            bce_weight=bce_weight,
            tversky_weight=tversky_weight,
            alpha=tversky_alpha,
            beta=tversky_beta,
            smooth=smooth,
            pos_weight=pos_weight,
        )

    raise ValueError(
        f"Unsupported loss_name={loss_name}. "
        "Supported: bce_dice, dice_tversky, focal_dice, focal_tversky, bce_tversky"
    )
