# scripts/evaluate/threshold_sweep_siim_smp_next_stage.py
# -*- coding: utf-8 -*-
"""
Threshold sweep for next-stage SIIM-ACR SMP models.

Compared with the original threshold_sweep_siim_smp.py, this script:
    1. supports decoder_attention_type in config/checkpoint
    2. reports image-level positive Dice and negative FP rate, which are more
       useful for selecting SIIM-ACR pneumothorax operating points
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset  # noqa: E402
from cxr_lesionseg.models import SMPModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Threshold sweep for next-stage SIIM-ACR SMP model")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-csv", type=str, default="threshold_sweep_next_stage.csv")

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_merged_config(config_path: str, ckpt: Dict) -> Dict:
    cfg: Dict = {}
    if isinstance(ckpt.get("config"), dict):
        cfg.update(ckpt["config"])
    with open(config_path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)
    cfg.update(file_cfg)
    return cfg


def compute_pixel_metrics(tp, fp, fn, tn, eps=1e-7):
    return {
        "dice": float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "specificity": float((tn + eps) / (tn + fp + eps)),
    }


def init_stats():
    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "num_images": 0.0,
        "num_positive": 0.0,
        "num_negative": 0.0,
        "sum_dice_all": 0.0,
        "sum_iou_all": 0.0,
        "sum_dice_positive": 0.0,
        "sum_iou_positive": 0.0,
        "sum_dice_negative": 0.0,
        "negative_false_positive": 0.0,
        "positive_detected": 0.0,
    }


def update_stats(stats: Dict[float, Dict[str, float]], thresholds, probs: torch.Tensor, masks: torch.Tensor):
    targets = (masks >= 0.5).float()
    targets_flat = targets.flatten(1)
    target_area = targets_flat.sum(dim=1)
    positive = target_area > 0
    negative = target_area == 0

    for th in thresholds:
        preds = (probs >= th).float()
        preds_flat_img = preds.flatten(1)
        preds_flat = preds.flatten()
        targets_flat_all = targets.flatten()

        s = stats[th]
        s["tp"] += (preds_flat * targets_flat_all).sum().item()
        s["fp"] += (preds_flat * (1.0 - targets_flat_all)).sum().item()
        s["fn"] += ((1.0 - preds_flat) * targets_flat_all).sum().item()
        s["tn"] += ((1.0 - preds_flat) * (1.0 - targets_flat_all)).sum().item()

        pred_area = preds_flat_img.sum(dim=1)
        intersection = (preds_flat_img * targets_flat).sum(dim=1)
        union = pred_area + target_area
        dice = torch.where(union > 0, (2.0 * intersection + 1e-7) / (union + 1e-7), torch.ones_like(union))
        iou = torch.where(union > 0, (intersection + 1e-7) / (pred_area + target_area - intersection + 1e-7), torch.ones_like(union))
        pred_positive = pred_area > 0

        s["num_images"] += float(targets.size(0))
        s["sum_dice_all"] += float(dice.sum().item())
        s["sum_iou_all"] += float(iou.sum().item())

        if positive.any():
            s["num_positive"] += float(positive.sum().item())
            s["sum_dice_positive"] += float(dice[positive].sum().item())
            s["sum_iou_positive"] += float(iou[positive].sum().item())
            s["positive_detected"] += float((positive & pred_positive).sum().item())

        if negative.any():
            s["num_negative"] += float(negative.sum().item())
            s["sum_dice_negative"] += float(dice[negative].sum().item())
            s["negative_false_positive"] += float((negative & pred_positive).sum().item())


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ckpt = safe_torch_load(args.checkpoint, device=device)
    cfg = load_merged_config(args.config, ckpt)

    dataset = SIIMACRDataset(
        processed_root=cfg["processed_root"],
        split_root=cfg["split_root"],
        split=args.split,
        image_size=int(cfg["image_size"]),
        in_channels=int(cfg["in_channels"]),
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(cfg["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = SMPModel(
        model_name=cfg["model_name"],
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=int(cfg["in_channels"]),
        classes=1,
        decoder_attention_type=cfg.get("decoder_attention_type", None),
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    stats = {th: init_stats() for th in args.thresholds}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits)
        update_stats(stats, args.thresholds, probs, masks)

    rows = []
    for th in args.thresholds:
        s = stats[th]
        m = compute_pixel_metrics(s["tp"], s["fp"], s["fn"], s["tn"])
        n_all = max(s["num_images"], 1.0)
        n_pos = max(s["num_positive"], 1.0)
        n_neg = max(s["num_negative"], 1.0)
        row = {
            "threshold": th,
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "mean_dice_all": s["sum_dice_all"] / n_all,
            "mean_iou_all": s["sum_iou_all"] / n_all,
            "mean_dice_positive": s["sum_dice_positive"] / n_pos,
            "mean_iou_positive": s["sum_iou_positive"] / n_pos,
            "mean_dice_negative": s["sum_dice_negative"] / n_neg,
            "negative_fp_rate": s["negative_false_positive"] / n_neg,
            "positive_image_recall": s["positive_detected"] / n_pos,
            "num_positive": s["num_positive"],
            "num_negative": s["num_negative"],
            "negative_false_positive": s["negative_false_positive"],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "tn": s["tn"],
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["mean_dice_positive"], reverse=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 120)
    print("Threshold sweep result sorted by mean_dice_positive")
    print("=" * 120)
    print(f"{'thr':>6} {'pix_dice':>9} {'pix_recall':>10} {'pos_dice':>10} {'neg_fp':>9} {'pos_img_rec':>11} {'mean_all':>10}")
    for row in rows:
        print(
            f"{row['threshold']:6.2f} "
            f"{row['dice']:9.4f} "
            f"{row['recall']:10.4f} "
            f"{row['mean_dice_positive']:10.4f} "
            f"{row['negative_fp_rate']:9.4f} "
            f"{row['positive_image_recall']:11.4f} "
            f"{row['mean_dice_all']:10.4f}"
        )
    print("=" * 120)
    print(f"Best positive threshold : {rows[0]['threshold']}")
    print(f"Best positive dice      : {rows[0]['mean_dice_positive']:.4f}")
    print(f"Saved to                : {output_csv}")
    print("=" * 120)


if __name__ == "__main__":
    main()
