# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.models import SMPModel


def parse_args():
    parser = argparse.ArgumentParser("Threshold sweep for SIIM-ACR SMP model")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output-csv", type=str, default="threshold_sweep.csv")

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)

    return parser.parse_args()


def compute_metrics(tp, fp, fn, tn, eps=1e-7):
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

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
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    stats = {
        th: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
        for th in args.thresholds
    }

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)
        targets = (masks >= 0.5).float()

        for th in args.thresholds:
            preds = (probs >= th).float()

            preds_flat = preds.flatten()
            targets_flat = targets.flatten()

            tp = (preds_flat * targets_flat).sum().item()
            fp = (preds_flat * (1.0 - targets_flat)).sum().item()
            fn = ((1.0 - preds_flat) * targets_flat).sum().item()
            tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

            stats[th]["tp"] += tp
            stats[th]["fp"] += fp
            stats[th]["fn"] += fn
            stats[th]["tn"] += tn

    rows = []

    for th in args.thresholds:
        s = stats[th]
        m = compute_metrics(s["tp"], s["fp"], s["fn"], s["tn"])

        row = {
            "threshold": th,
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "tn": s["tn"],
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["dice"], reverse=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("Threshold sweep result")
    print("=" * 100)
    print(f"{'threshold':>10} {'dice':>10} {'iou':>10} {'precision':>10} {'recall':>10} {'specificity':>12}")

    for row in rows:
        print(
            f"{row['threshold']:10.2f} "
            f"{row['dice']:10.4f} "
            f"{row['iou']:10.4f} "
            f"{row['precision']:10.4f} "
            f"{row['recall']:10.4f} "
            f"{row['specificity']:12.4f}"
        )

    print("=" * 100)
    print(f"Best threshold : {rows[0]['threshold']}")
    print(f"Best dice      : {rows[0]['dice']:.4f}")
    print(f"Saved to       : {output_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
