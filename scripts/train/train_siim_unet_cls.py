# scripts/train/train_siim_unet_cls.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.losses import BCEDiceLoss
from cxr_lesionseg.models import UNetWithClassificationHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train U-Net with classification head on SIIM-ACR."
    )

    parser.add_argument("--processed-root", type=str, default="data/processed/siim_acr_pneumothorax")
    parser.add_argument("--split-root", type=str, default="data/splits/siim_acr_pneumothorax")
    parser.add_argument("--output-root", type=str, default="outputs/experiments/siim_acr_unet_cls")

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--cls-dropout", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)

    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--pos-weight", type=float, default=2.0)

    parser.add_argument(
        "--cls-weight",
        type=float,
        default=0.2,
        help="Weight of image-level classification loss.",
    )
    parser.add_argument(
        "--cls-pos-weight",
        type=float,
        default=None,
        help="Positive class weight for image-level BCE loss.",
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cls-threshold", type=float, default=0.5)

    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--positive-sample-ratio", type=float, default=0.35)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")

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
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def load_image_label_map(processed_root: str) -> Dict[str, int]:
    summary_path = Path(processed_root) / "mask_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"mask_summary.csv not found: {summary_path}")

    label_map: Dict[str, int] = {}

    # 注意：mask_summary.csv 可能带 UTF-8 BOM，
    # 所以这里必须使用 utf-8-sig，否则第一列会变成 "\ufeffImageId"。
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(f"Empty csv file: {summary_path}")

        required_columns = {"ImageId", "has_mask"}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise KeyError(
                f"Missing columns in {summary_path}: {missing_columns}. "
                f"Actual columns: {reader.fieldnames}"
            )

        for row in reader:
            image_id = str(row["ImageId"]).strip()
            has_mask = int(row["has_mask"])
            label_map[image_id] = has_mask

    print("-" * 80)
    print("[Image Label Map]")
    print(f"summary_path : {summary_path}")
    print(f"num_images   : {len(label_map)}")
    print("-" * 80)

    return label_map


def build_weighted_sampler(
        dataset: SIIMACRDataset,
        processed_root: str,
        positive_sample_ratio: float,
) -> WeightedRandomSampler:
    label_map = load_image_label_map(processed_root)

    labels = np.array(
        [int(label_map.get(image_id, 0)) for image_id in dataset.image_ids],
        dtype=np.int64,
    )

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

    print("-" * 80)
    print("[Balanced Sampling]")
    print(f"positive images       : {pos_count}")
    print(f"negative images       : {neg_count}")
    print(f"positive_sample_ratio : {positive_sample_ratio}")
    print("-" * 80)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


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


def make_cls_targets(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: [B, 1, H, W]
    return: [B, 1], 1 means positive image.
    """
    return (masks.flatten(1).sum(dim=1) > 0).float().unsqueeze(1)


@torch.no_grad()
def segmentation_metrics_from_logits(
        seg_logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float,
        cls_logits: torch.Tensor | None = None,
        cls_threshold: float = 0.5,
        eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(seg_logits)
    preds = (probs >= threshold).float()
    targets = (masks >= 0.5).float()

    if cls_logits is not None:
        cls_probs = torch.sigmoid(cls_logits)
        keep = (cls_probs >= cls_threshold).float().view(-1, 1, 1, 1)
        preds = preds * keep

    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1.0 - targets_flat)).sum()
    fn = ((1.0 - preds_flat) * targets_flat).sum()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum()

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


@torch.no_grad()
def classification_metrics_from_logits(
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        cls_threshold: float,
        eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(cls_logits)
    preds = (probs >= cls_threshold).float()
    targets = (cls_targets >= 0.5).float()

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()

    acc = ((preds == targets).float()).mean()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "cls_acc": float(acc.item()),
        "cls_precision": float(precision.item()),
        "cls_recall": float(recall.item()),
        "cls_specificity": float(specificity.item()),
    }


def update_average(avg: Dict[str, float], current: Dict[str, float], n: int) -> None:
    for key, value in current.items():
        if key not in avg:
            avg[key] = 0.0
        avg[key] += value * n


def finalize_average(avg: Dict[str, float], total: int) -> Dict[str, float]:
    return {key: value / max(total, 1) for key, value in avg.items()}


def autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)

    return nullcontext()


def build_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        seg_criterion: torch.nn.Module,
        cls_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler,
        use_amp: bool,
        cls_weight: float,
        threshold: float,
        cls_threshold: float,
) -> Dict[str, float]:
    model.train()

    total_samples = 0
    avg: Dict[str, float] = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        cls_targets = make_cls_targets(masks)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            outputs = model(images)

            seg_logits = outputs["seg_logits"]
            cls_logits = outputs["cls_logits"]

            seg_loss = seg_criterion(seg_logits, masks)
            cls_loss = cls_criterion(cls_logits, cls_targets)
            loss = seg_loss + cls_weight * cls_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)

        seg_metrics = segmentation_metrics_from_logits(
            seg_logits.detach(),
            masks,
            threshold=threshold,
        )
        cls_metrics = classification_metrics_from_logits(
            cls_logits.detach(),
            cls_targets,
            cls_threshold=cls_threshold,
        )

        metrics = {
            "loss": float(loss.item()),
            "seg_loss": float(seg_loss.item()),
            "cls_loss": float(cls_loss.item()),
            **seg_metrics,
            **cls_metrics,
        }

        update_average(avg, metrics, batch_size)
        total_samples += batch_size

    return finalize_average(avg, total_samples)


@torch.no_grad()
def validate_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        seg_criterion: torch.nn.Module,
        cls_criterion: torch.nn.Module,
        device: torch.device,
        use_amp: bool,
        cls_weight: float,
        threshold: float,
        cls_threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_samples = 0
    avg: Dict[str, float] = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        cls_targets = make_cls_targets(masks)

        with autocast_context(device, use_amp):
            outputs = model(images)

            seg_logits = outputs["seg_logits"]
            cls_logits = outputs["cls_logits"]

            seg_loss = seg_criterion(seg_logits, masks)
            cls_loss = cls_criterion(cls_logits, cls_targets)
            loss = seg_loss + cls_weight * cls_loss

        raw_metrics = segmentation_metrics_from_logits(
            seg_logits,
            masks,
            threshold=threshold,
        )

        gated_metrics = segmentation_metrics_from_logits(
            seg_logits,
            masks,
            threshold=threshold,
            cls_logits=cls_logits,
            cls_threshold=cls_threshold,
        )

        cls_metrics = classification_metrics_from_logits(
            cls_logits,
            cls_targets,
            cls_threshold=cls_threshold,
        )

        batch_size = images.size(0)

        metrics = {
            "loss": float(loss.item()),
            "seg_loss": float(seg_loss.item()),
            "cls_loss": float(cls_loss.item()),

            "raw_dice": raw_metrics["dice"],
            "raw_iou": raw_metrics["iou"],
            "raw_precision": raw_metrics["precision"],
            "raw_recall": raw_metrics["recall"],
            "raw_specificity": raw_metrics["specificity"],

            "gated_dice": gated_metrics["dice"],
            "gated_iou": gated_metrics["iou"],
            "gated_precision": gated_metrics["precision"],
            "gated_recall": gated_metrics["recall"],
            "gated_specificity": gated_metrics["specificity"],

            **cls_metrics,
        }

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
        best_score: float,
        args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
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
    print("[Train SIIM-ACR U-Net + Classification Head]")
    print(f"device          : {device}")
    print(f"run_dir         : {run_dir}")
    print(f"image_size      : {args.image_size}")
    print(f"batch_size      : {args.batch_size}")
    print(f"epochs          : {args.epochs}")
    print(f"amp             : {args.amp}")
    print(f"cls_weight      : {args.cls_weight}")
    print(f"threshold       : {args.threshold}")
    print(f"cls_threshold   : {args.cls_threshold}")
    print("=" * 80)

    train_loader, val_loader = build_loaders(args)

    model = UNetWithClassificationHead(
        in_channels=args.in_channels,
        out_channels=1,
        base_channels=args.base_channels,
        cls_dropout=args.cls_dropout,
    ).to(device)

    seg_criterion = BCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
    )

    if args.cls_pos_weight is None:
        cls_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        cls_pos_weight = torch.tensor([args.cls_pos_weight], dtype=torch.float32, device=device)
        cls_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)

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
    scaler = build_scaler(use_amp)

    metrics_path = run_dir / "logs" / "metrics.csv"

    best_raw_score = -1.0
    best_gated_score = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            seg_criterion=seg_criterion,
            cls_criterion=cls_criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            cls_weight=args.cls_weight,
            threshold=args.threshold,
            cls_threshold=args.cls_threshold,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            seg_criterion=seg_criterion,
            cls_criterion=cls_criterion,
            device=device,
            use_amp=use_amp,
            cls_weight=args.cls_weight,
            threshold=args.threshold,
            cls_threshold=args.cls_threshold,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,

            "train_loss": train_metrics["loss"],
            "train_seg_loss": train_metrics["seg_loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_specificity": train_metrics["specificity"],
            "train_cls_acc": train_metrics["cls_acc"],
            "train_cls_precision": train_metrics["cls_precision"],
            "train_cls_recall": train_metrics["cls_recall"],
            "train_cls_specificity": train_metrics["cls_specificity"],

            "val_loss": val_metrics["loss"],
            "val_seg_loss": val_metrics["seg_loss"],
            "val_cls_loss": val_metrics["cls_loss"],

            "val_raw_dice": val_metrics["raw_dice"],
            "val_raw_iou": val_metrics["raw_iou"],
            "val_raw_precision": val_metrics["raw_precision"],
            "val_raw_recall": val_metrics["raw_recall"],
            "val_raw_specificity": val_metrics["raw_specificity"],

            "val_gated_dice": val_metrics["gated_dice"],
            "val_gated_iou": val_metrics["gated_iou"],
            "val_gated_precision": val_metrics["gated_precision"],
            "val_gated_recall": val_metrics["gated_recall"],
            "val_gated_specificity": val_metrics["gated_specificity"],

            "val_cls_acc": val_metrics["cls_acc"],
            "val_cls_precision": val_metrics["cls_precision"],
            "val_cls_recall": val_metrics["cls_recall"],
            "val_cls_specificity": val_metrics["cls_specificity"],
        }

        append_metrics(metrics_path, row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_raw_dice={val_metrics['raw_dice']:.4f} "
            f"val_gated_dice={val_metrics['gated_dice']:.4f} "
            f"val_cls_acc={val_metrics['cls_acc']:.4f} "
            f"val_cls_recall={val_metrics['cls_recall']:.4f} "
            f"val_cls_specificity={val_metrics['cls_specificity']:.4f}"
        )

        save_checkpoint(
            save_path=run_dir / "checkpoints" / "last_model.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=max(best_raw_score, best_gated_score),
            args=args,
        )

        current_raw_score = val_metrics["raw_dice"]
        current_gated_score = val_metrics["gated_dice"]

        if current_raw_score > best_raw_score:
            best_raw_score = current_raw_score

            save_checkpoint(
                save_path=run_dir / "checkpoints" / "best_raw_model.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_raw_score,
                args=args,
            )

            print(f"  -> Saved best raw model. best_val_raw_dice={best_raw_score:.4f}")

        if current_gated_score > best_gated_score:
            best_gated_score = current_gated_score

            save_checkpoint(
                save_path=run_dir / "checkpoints" / "best_gated_model.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_gated_score,
                args=args,
            )

        print(f"  -> Saved best gated model. best_val_gated_dice={best_gated_score:.4f}")

    print("=" * 80)
    print("[Done]")
    print(f"Best val raw dice   : {best_raw_score:.4f}")
    print(f"Best val gated dice : {best_gated_score:.4f}")
    print(f"Run dir             : {run_dir}")
    print(f"Metrics             : {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()