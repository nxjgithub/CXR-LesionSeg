# src/cxr_lesionseg/metrics/segmentation_metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def binary_segmentation_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute binary segmentation metrics.

    Args:
        logits: [B, 1, H, W]
        targets: [B, 1, H, W]

    Returns:
        dice, iou, precision, recall, specificity
    """

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    preds = preds.flatten()
    targets = targets.flatten()

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "specificity": float(specificity.item()),
    }