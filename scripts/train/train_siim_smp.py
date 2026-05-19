# scripts/train/train_siim_smp.py
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
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.losses import BCEDiceLoss
from cxr_lesionseg.models import SMPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SMP segmentation model on SIIM-ACR."
    )

    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed/siim_acr_pneumothorax",
    )
    parser.add_argument(
        "--split-root",
        type=str,
        default="data/splits/siim_acr_pneumothorax",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/experiments/siim_acr_smp",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="unetplusplus",
        choices=["unet", "unetplusplus", "deeplabv3plus"],
    )
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Use imagenet for pretrained encoder, or none for random init.",
    )

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)

    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--pos-weight", type=float, default=2.0)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--positive-sample-ratio", type=float, default=0.35)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


def normalize_encoder_weights(value: str | None) -> str | None:
    if value is None:
        return None

    value = str(value).strip()

    if value.lower() in ["none", "null", "false", ""]:
        return None

    return value


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
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
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


def autocast_context(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)

    return nullcontext()


def build_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def update_confusion(
        logits: torch.Tensor,
        masks: torch.Tensor,
        threshold: float,
        stats: Dict[str, float],
) -> None:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (masks >= 0.5).float()

    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    stats["tn"] += tn


def compute_metrics_from_confusion(stats: Dict[str, float], eps: float = 1e-7) -> Dict[str, float]:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]

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


def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scaler,
        use_amp: bool,
        threshold: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    stats = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
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
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        update_confusion(
            logits=logits.detach(),
            masks=masks,
            threshold=threshold,
            stats=stats,
        )

    metrics = compute_metrics_from_confusion(stats)
    metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics


@torch.no_grad()
def validate_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        use_amp: bool,
        threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    stats = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        update_confusion(
            logits=logits,
            masks=masks,
            threshold=threshold,
            stats=stats,
        )

    metrics = compute_metrics_from_confusion(stats)
    metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics


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

    args.encoder_weights = normalize_encoder_weights(args.encoder_weights)

    set_seed(args.seed)

    device = get_device(args.device)
    run_dir = create_run_dir(args.output_root)
    save_config(args, run_dir)

    print("=" * 80)
    print("[Train SIIM-ACR SMP Model]")
    print(f"device          : {device}")
    print(f"run_dir         : {run_dir}")
    print(f"model_name      : {args.model_name}")
    print(f"encoder_name    : {args.encoder_name}")
    print(f"encoder_weights : {args.encoder_weights}")
    print(f"in_channels     : {args.in_channels}")
    print(f"image_size      : {args.image_size}")
    print(f"batch_size      : {args.batch_size}")
    print(f"epochs          : {args.epochs}")
    print(f"amp             : {args.amp}")
    print("=" * 80)

    train_loader, val_loader = build_loaders(args)

    model = SMPModel(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        classes=1,
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
    scaler = build_scaler(use_amp)

    metrics_path = run_dir / "logs" / "metrics.csv"

    best_score = -1.0

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
            use_amp=use_amp,
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
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        save_checkpoint(
            save_path=run_dir / "checkpoints" / "last_model.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_score=best_score,
            args=args,
        )

        current_score = val_metrics["dice"]

        if current_score > best_score:
            best_score = current_score

            save_checkpoint(
                save_path=run_dir / "checkpoints" / "best_model.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )

            print(f"  -> Saved best model. best_val_dice={best_score:.4f}")

    print("=" * 80)
    print("[Done]")
    print(f"Best val dice : {best_score:.4f}")
    print(f"Run dir       : {run_dir}")
    print(f"Metrics       : {metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()