# scripts/evaluate/threshold_sweep_siim_unet.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep for SIIM-ACR U-Net.")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processed-root", type=str, default="data/processed/siim_acr_pneumothorax")
    parser.add_argument("--split-root", type=str, default="data/splits/siim_acr_pneumothorax")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)

    parser.add_argument(
        "--min-pred-area",
        type=int,
        default=30,
        help="For negative images, predictions smaller than this area are ignored.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Where to save threshold sweep results.",
    )

    return parser.parse_args()


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    in_channels = int(config.get("in_channels", 1))
    base_channels = int(config.get("base_channels", 32))

    model = UNet(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("=" * 80)
    print("[Checkpoint]")
    print(f"path          : {checkpoint_path}")
    print(f"epoch         : {ckpt.get('epoch')}")
    print(f"best_dice     : {ckpt.get('best_dice')}")
    print(f"in_channels   : {in_channels}")
    print(f"base_channels : {base_channels}")
    print("=" * 80)

    return model


def build_loader(args: argparse.Namespace) -> DataLoader:
    dataset = SIIMACRDataset(
        processed_root=args.processed_root,
        split_root=args.split_root,
        split=args.split,
        image_size=args.image_size,
        in_channels=1,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    print("=" * 80)
    print("[Dataset]")
    print(f"split  : {args.split}")
    print(f"length : {len(dataset)}")
    print("=" * 80)

    return loader


@torch.no_grad()
def collect_probs_and_masks(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
) -> Dict[str, torch.Tensor]:
    probs_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        probs = torch.sigmoid(logits)

        probs_list.append(probs.cpu())
        masks_list.append(masks.cpu())

        if step % 20 == 0:
            print(f"[Collect] step={step}")

    probs_all = torch.cat(probs_list, dim=0)
    masks_all = torch.cat(masks_list, dim=0)

    return {
        "probs": probs_all,
        "masks": masks_all,
    }


def compute_metrics_at_threshold(
        probs: torch.Tensor,
        masks: torch.Tensor,
        threshold: float,
        min_pred_area: int = 30,
        eps: float = 1e-7,
) -> Dict[str, float]:
    preds = (probs >= threshold).float()
    targets = (masks >= 0.5).float()

    # global pixel-level metrics
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1.0 - targets_flat)).sum()
    fn = ((1.0 - preds_flat) * targets_flat).sum()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum()

    global_dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    global_iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    # image-level statistics
    b = preds.shape[0]
    preds_img = preds.view(b, -1)
    targets_img = targets.view(b, -1)

    pred_area = preds_img.sum(dim=1)
    target_area = targets_img.sum(dim=1)

    pos_mask = target_area > 0
    neg_mask = target_area == 0

    intersection = (preds_img * targets_img).sum(dim=1)
    dice_per_image = (2.0 * intersection + eps) / (pred_area + target_area + eps)

    # empty target + empty pred is considered dice=1 for all-image mean
    empty_correct = (target_area == 0) & (pred_area == 0)
    dice_per_image = torch.where(empty_correct, torch.ones_like(dice_per_image), dice_per_image)

    if pos_mask.sum() > 0:
        positive_mean_dice = dice_per_image[pos_mask].mean()
    else:
        positive_mean_dice = torch.tensor(0.0)

    all_mean_dice = dice_per_image.mean()

    if neg_mask.sum() > 0:
        neg_pred_area = pred_area[neg_mask]
        negative_fp_rate = (neg_pred_area >= min_pred_area).float().mean()
    else:
        negative_fp_rate = torch.tensor(0.0)

    return {
        "threshold": float(threshold),
        "global_dice": float(global_dice.item()),
        "global_iou": float(global_iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "specificity": float(specificity.item()),
        "positive_mean_dice": float(positive_mean_dice.item()),
        "all_mean_dice": float(all_mean_dice.item()),
        "negative_fp_rate": float(negative_fp_rate.item()),
        "positive_images": int(pos_mask.sum().item()),
        "negative_images": int(neg_mask.sum().item()),
    }


def make_thresholds(start: float, end: float, step: float) -> List[float]:
    thresholds = []
    value = start

    while value <= end + 1e-9:
        thresholds.append(round(value, 4))
        value += step

    return thresholds


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    model = load_checkpoint_model(args.checkpoint, device=device)
    loader = build_loader(args)

    data = collect_probs_and_masks(model, loader, device=device)
    probs = data["probs"]
    masks = data["masks"]

    thresholds = make_thresholds(
        start=args.threshold_start,
        end=args.threshold_end,
        step=args.threshold_step,
    )

    rows = []

    print("=" * 80)
    print("[Threshold Sweep]")
    print(f"thresholds: {thresholds}")
    print("=" * 80)

    for threshold in thresholds:
        row = compute_metrics_at_threshold(
            probs=probs,
            masks=masks,
            threshold=threshold,
            min_pred_area=args.min_pred_area,
        )
        rows.append(row)

        print(
            f"thr={threshold:.2f} | "
            f"global_dice={row['global_dice']:.4f} | "
            f"pos_dice={row['positive_mean_dice']:.4f} | "
            f"precision={row['precision']:.4f} | "
            f"recall={row['recall']:.4f} | "
            f"neg_fp_rate={row['negative_fp_rate']:.4f}"
        )

    best_global = max(rows, key=lambda x: x["global_dice"])
    best_positive = max(rows, key=lambda x: x["positive_mean_dice"])

    print("=" * 80)
    print("[Best by global_dice]")
    print(best_global)
    print("-" * 80)
    print("[Best by positive_mean_dice]")
    print(best_positive)
    print("=" * 80)

    if args.output_csv is None:
        ckpt_path = Path(args.checkpoint)
        output_csv = ckpt_path.parent.parent / "logs" / "threshold_sweep.csv"
    else:
        output_csv = Path(args.output_csv)

    save_csv(output_csv, rows)
    print(f"[Saved] {output_csv}")


if __name__ == "__main__":
    main()