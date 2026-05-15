# scripts/train/train_siim_unet.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.losses import BCEDiceLoss
from cxr_lesionseg.metrics import binary_segmentation_metrics
from cxr_lesionseg.models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net baseline on SIIM-ACR.")

    parser.add_argument("--processed-root", type=str, default="data/processed/siim_acr_pneumothorax")
    parser.add_argument("--split-root", type=str, default="data/splits/siim_acr_pneumothorax")

    parser.add_argument("--output-root", type=str, default="outputs/experiments/siim_acr_unet_baseline")

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")

    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use weighted sampling to increase positive samples in each epoch.",
    )

    parser.add_argument(
        "--positive-sample-ratio",
        type=float,
        default=0.5,
        help="Expected positive sample ratio when balanced sampling is enabled.",
    )

    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive pixel weight for BCEWithLogitsLoss.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_run_dir(output_root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    return run_dir


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

def build_weighted_sampler(
        dataset: SIIMACRDataset,
        processed_root: str,
        positive_sample_ratio: float = 0.5,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler according to image-level positive / negative labels.

    This does not change the validation set.
    It only makes positive training images appear more frequently.
    """
    summary_path = Path(processed_root) / "mask_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"mask_summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)
    df["ImageId"] = df["ImageId"].astype(str)

    has_mask_map = dict(zip(df["ImageId"], df["has_mask"]))

    labels = []
    for image_id in dataset.image_ids:
        labels.append(int(has_mask_map.get(image_id, 0)))

    labels = np.asarray(labels, dtype=np.int64)

    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())

    if pos_count == 0 or neg_count == 0:
        raise RuntimeError(
            f"Cannot build weighted sampler. pos_count={pos_count}, neg_count={neg_count}"
        )

    pos_weight = positive_sample_ratio / pos_count
    neg_weight = (1.0 - positive_sample_ratio) / neg_count

    weights = np.where(labels == 1, pos_weight, neg_weight)
    weights = torch.as_tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    print("-" * 80)
    print("[Balanced Sampling]")
    print(f"positive images       : {pos_count}")
    print(f"negative images       : {neg_count}")
    print(f"positive_sample_ratio : {positive_sample_ratio}")
    print("-" * 80)

    return sampler


def build_loaders(args: argparse.Namespace):
    train_dataset = SIIMACRDataset(
        processed_root=args.processed_root,
        split_root=args.split_root,
        split="train",
        image_size=args.image_size,
        in_channels=args.in_channels,
        normalize=True,
    )

    val_dataset = SIIMACRDataset(
        processed_root=args.processed_root,
        split_root=args.split_root,
        split="val",
        image_size=args.image_size,
        in_channels=args.in_channels,
        normalize=True,
    )

    if args.balanced_sampling:
        train_sampler = build_weighted_sampler(
            dataset=train_dataset,
            processed_root=args.processed_root,
            positive_sample_ratio=args.positive_sample_ratio,
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


def update_average(avg: Dict[str, float], current: Dict[str, float], n: int) -> None:
    for key, value in current.items():
        avg[key] += value * n


def finalize_average(avg: Dict[str, float], total: int) -> Dict[str, float]:
    return {key: value / max(total, 1) for key, value in avg.items()}


def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        use_amp: bool,
        threshold: float,
) -> Dict[str, float]:
    model.train()

    total_samples = 0

    avg = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        metrics = binary_segmentation_metrics(logits.detach(), masks, threshold=threshold)
        metrics["loss"] = float(loss.item())

        update_average(avg, metrics, batch_size)
        total_samples += batch_size

    return finalize_average(avg, total_samples)


@torch.no_grad()
def validate_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_samples = 0

    avg = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
    }

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        metrics = binary_segmentation_metrics(logits, masks, threshold=threshold)
        metrics["loss"] = float(loss.item())

        update_average(avg, metrics, batch_size)
        total_samples += batch_size

    return finalize_average(avg, total_samples)


def append_metrics(metrics_path: Path, row: Dict[str, float]) -> None:
    is_new = not metrics_path.exists()

    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if is_new:
            writer.writeheader()

        writer.writerow(row)


def save_checkpoint(
        save_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_dice: float,
        args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        },
        save_path,
    )


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = get_device(args.device)
    run_dir = create_run_dir(args.output_root)
    save_config(args, run_dir)

    print("=" * 80)
    print("[Train SIIM-ACR U-Net Baseline]")
    print(f"device      : {device}")
    print(f"run_dir     : {run_dir}")
    print(f"image_size  : {args.image_size}")
    print(f"batch_size  : {args.batch_size}")
    print(f"epochs      : {args.epochs}")
    print(f"amp         : {args.amp}")
    print("=" * 80)

    train_loader, val_loader = build_loaders(args)

    model = UNet(
        in_channels=args.in_channels,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = run_dir / "logs" / "metrics.csv"

    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            threshold=args.threshold,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_specificity": train_metrics["specificity"],

            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
        }

        append_metrics(metrics_path, row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        last_ckpt_path = run_dir / "checkpoints" / "last_model.pth"
        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_dice=best_dice,
            args=args,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            best_ckpt_path = run_dir / "checkpoints" / "best_model.pth"
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dice=best_dice,
                args=args,
            )

            print(f"  -> Saved best model. best_val_dice={best_dice:.4f}")

    print("=" * 80)
    print("[Done]")
    print(f"Best val dice : {best_dice:.4f}")
    print(f"Run dir       : {run_dir}")
    print(f"Metrics       : {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()