# scripts/train/train_siim_smp_next_stage.py
# -*- coding: utf-8 -*-
"""
Next-stage SIIM-ACR SMP training script.

Main additions over train_siim_smp.py:
    1. advanced losses: dice_tversky / focal_dice / focal_tversky / bce_tversky
    2. early stopping
    3. optional CXR-friendly Albumentations augmentation
    4. optional scSE decoder attention for U-Net / U-Net++
    5. extra validation metrics: image-level positive Dice, negative FP rate,
       positive image recall

Before using this script, copy these two replacement files:
    cp segmentation_losses_next_stage.py src/cxr_lesionseg/losses/segmentation_losses.py
    cp smp_models_next_stage.py src/cxr_lesionseg/models/smp_models.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset  # noqa: E402
from cxr_lesionseg.losses.segmentation_losses import build_segmentation_loss  # noqa: E402
from cxr_lesionseg.models import SMPModel  # noqa: E402


class SIIMACRAugDataset(SIIMACRDataset):
    """SIIM dataset with synchronized image/mask Albumentations transform."""

    def __init__(self, *args, transform=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_id = self.image_ids[index]
        image_path = self._find_image_path(self.image_dir, image_id)

        image = self._load_image(image_path)
        image = self._resize_image(image, self.image_size)
        image_arr = np.asarray(image)

        sample: Dict[str, object] = {
            "image_id": image_id,
            "image_path": str(image_path),
        }

        if self.split != "test":
            mask_path = self.mask_dir / f"{image_id}.png"
            mask = self._load_mask(mask_path)
            mask = self._resize_mask(mask, self.image_size)
            mask_arr = (np.asarray(mask) > 127).astype(np.uint8)

            if self.transform is not None:
                augmented = self.transform(image=image_arr, mask=mask_arr)
                image_arr = augmented["image"]
                mask_arr = augmented["mask"]

            sample["image"] = self._image_array_to_tensor(image_arr)
            sample["mask"] = self._mask_array_to_tensor(mask_arr)
            sample["mask_path"] = str(mask_path)
        else:
            if self.transform is not None:
                augmented = self.transform(image=image_arr)
                image_arr = augmented["image"]
            sample["image"] = self._image_array_to_tensor(image_arr)

        return sample

    def _image_array_to_tensor(self, image_arr: np.ndarray) -> torch.Tensor:
        arr = image_arr.astype(np.float32) / 255.0

        if self.in_channels == 1:
            if arr.ndim == 3:
                arr = arr[..., 0]
            arr = arr[None, :, :]
        else:
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            arr = arr.transpose(2, 0, 1)

        if self.normalize:
            arr = (arr - self.mean) / self.std

        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    @staticmethod
    def _mask_array_to_tensor(mask_arr: np.ndarray) -> torch.Tensor:
        arr = (mask_arr > 0).astype(np.float32)
        arr = arr[None, :, :]
        return torch.from_numpy(np.ascontiguousarray(arr)).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train next-stage SMP segmentation model on SIIM-ACR.")

    parser.add_argument("--processed-root", type=str, default="data/processed/siim_acr_pneumothorax")
    parser.add_argument("--split-root", type=str, default="data/splits/siim_acr_pneumothorax")
    parser.add_argument("--output-root", type=str, default="outputs/experiments/siim_acr_smp_next_stage")

    parser.add_argument("--model-name", type=str, default="unetplusplus", choices=["unet", "unetplusplus", "deeplabv3plus"])
    parser.add_argument("--encoder-name", type=str, default="timm-efficientnet-b3")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--decoder-attention-type", type=str, default="none", choices=["none", "scse"])

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument(
        "--loss-name",
        type=str,
        default="dice_tversky",
        choices=["bce_dice", "dice_tversky", "focal_dice", "focal_tversky", "bce_tversky"],
    )
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--tversky-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=2.5)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-tversky-gamma", type=float, default=0.75)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fp-constraint", type=float, default=0.10, help="Save best_fp_constrained_model when val_negative_fp_rate <= this value.")

    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--positive-sample-ratio", type=float, default=0.45)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--aug-strength", type=str, default="light", choices=["light", "medium"])

    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_dice",
        choices=["val_dice", "val_mean_dice_positive", "val_mean_dice_all", "val_recall"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


def normalize_encoder_weights(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in {"none", "null", "false", ""}:
        return None
    return value


def normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"", "none", "null", "false", "0"}:
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


def build_train_transform(strength: str = "light"):
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "--augment requires albumentations. Install it with: pip install albumentations"
        ) from exc

    if strength == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=5, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.35),
            A.RandomGamma(gamma_limit=(85, 115), p=0.25),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
            A.GaussianBlur(blur_limit=(3, 3), p=0.10),
        ])

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=7, border_mode=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.45),
        A.RandomGamma(gamma_limit=(80, 120), p=0.35),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.20),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.15),
    ])


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
            raise KeyError(f"Missing columns in {summary_path}: {missing_columns}. Actual columns: {reader.fieldnames}")
        for row in reader:
            label_map[str(row["ImageId"]).strip()] = int(row["has_mask"])
    return label_map


def build_weighted_sampler(dataset: SIIMACRDataset, processed_root: str, positive_sample_ratio: float) -> WeightedRandomSampler:
    label_map = load_image_label_map(processed_root)
    labels = np.array([int(label_map.get(image_id, 0)) for image_id in dataset.image_ids], dtype=np.int64)

    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())
    if pos_count == 0 or neg_count == 0:
        raise RuntimeError(f"Cannot build weighted sampler. pos_count={pos_count}, neg_count={neg_count}")

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

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_transform = build_train_transform(args.aug_strength) if args.augment else None

    dataset_cls = SIIMACRAugDataset if args.augment else SIIMACRDataset

    train_dataset = dataset_cls(
        processed_root=args.processed_root,
        split_root=args.split_root,
        split="train",
        image_size=args.image_size,
        in_channels=args.in_channels,
        normalize=True,
        transform=train_transform,
    ) if args.augment else dataset_cls(
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
        train_sampler = build_weighted_sampler(train_dataset, args.processed_root, args.positive_sample_ratio)
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


def update_confusion(logits: torch.Tensor, masks: torch.Tensor, threshold: float, stats: Dict[str, float]) -> None:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (masks >= 0.5).float()

    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    stats["tp"] += (preds_flat * targets_flat).sum().item()
    stats["fp"] += (preds_flat * (1.0 - targets_flat)).sum().item()
    stats["fn"] += ((1.0 - preds_flat) * targets_flat).sum().item()
    stats["tn"] += ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()


def compute_metrics_from_confusion(stats: Dict[str, float], eps: float = 1e-7) -> Dict[str, float]:
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    return {
        "dice": float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "specificity": float((tn + eps) / (tn + fp + eps)),
    }


def update_image_level_metrics(logits: torch.Tensor, masks: torch.Tensor, threshold: float, image_stats: Dict[str, float]) -> None:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float().flatten(1)
    targets = (masks >= 0.5).float().flatten(1)

    pred_area = preds.sum(dim=1)
    target_area = targets.sum(dim=1)
    intersection = (preds * targets).sum(dim=1)
    union = pred_area + target_area

    dice = torch.where(
        union > 0,
        (2.0 * intersection + 1e-7) / (union + 1e-7),
        torch.ones_like(union),
    )
    iou = torch.where(
        union > 0,
        (intersection + 1e-7) / (pred_area + target_area - intersection + 1e-7),
        torch.ones_like(union),
    )

    positive = target_area > 0
    negative = target_area == 0
    pred_positive = pred_area > 0

    image_stats["num_images"] += float(targets.size(0))
    image_stats["sum_dice_all"] += float(dice.sum().item())
    image_stats["sum_iou_all"] += float(iou.sum().item())

    if positive.any():
        image_stats["num_positive"] += float(positive.sum().item())
        image_stats["sum_dice_positive"] += float(dice[positive].sum().item())
        image_stats["sum_iou_positive"] += float(iou[positive].sum().item())
        image_stats["positive_detected"] += float((positive & pred_positive).sum().item())

    if negative.any():
        image_stats["num_negative"] += float(negative.sum().item())
        image_stats["sum_dice_negative"] += float(dice[negative].sum().item())
        image_stats["negative_false_positive"] += float((negative & pred_positive).sum().item())


def finalize_image_level_metrics(image_stats: Dict[str, float]) -> Dict[str, float]:
    n_all = max(image_stats["num_images"], 1.0)
    n_pos = max(image_stats["num_positive"], 1.0)
    n_neg = max(image_stats["num_negative"], 1.0)

    return {
        "mean_dice_all": image_stats["sum_dice_all"] / n_all,
        "mean_iou_all": image_stats["sum_iou_all"] / n_all,
        "mean_dice_positive": image_stats["sum_dice_positive"] / n_pos,
        "mean_iou_positive": image_stats["sum_iou_positive"] / n_pos,
        "mean_dice_negative": image_stats["sum_dice_negative"] / n_neg,
        "negative_fp_rate": image_stats["negative_false_positive"] / n_neg,
        "positive_image_recall": image_stats["positive_detected"] / n_pos,
        "num_positive": image_stats["num_positive"],
        "num_negative": image_stats["num_negative"],
        "negative_false_positive": image_stats["negative_false_positive"],
    }


def init_image_stats() -> Dict[str, float]:
    return {
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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp: bool, threshold: float) -> Dict[str, float]:
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
        update_confusion(logits.detach(), masks, threshold, stats)

    metrics = compute_metrics_from_confusion(stats)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, use_amp: bool, threshold: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    stats = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    image_stats = init_image_stats()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        update_confusion(logits, masks, threshold, stats)
        update_image_level_metrics(logits, masks, threshold, image_stats)

    metrics = compute_metrics_from_confusion(stats)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics.update(finalize_image_level_metrics(image_stats))
    return metrics


def append_metrics(metrics_path: Path, row: Dict[str, float]) -> None:
    is_new = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(save_path: Path, model, optimizer, scheduler, epoch: int, best_score: float, args: argparse.Namespace) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": vars(args),
        },
        save_path,
    )


def main() -> None:
    args = parse_args()
    args.encoder_weights = normalize_encoder_weights(args.encoder_weights)
    args.decoder_attention_type = normalize_optional_string(args.decoder_attention_type)

    set_seed(args.seed)
    device = get_device(args.device)
    run_dir = create_run_dir(args.output_root)
    save_config(args, run_dir)

    print("=" * 100)
    print("[Train SIIM-ACR SMP Next Stage]")
    print(f"device                 : {device}")
    print(f"run_dir                : {run_dir}")
    print(f"model_name             : {args.model_name}")
    print(f"encoder_name           : {args.encoder_name}")
    print(f"encoder_weights        : {args.encoder_weights}")
    print(f"decoder_attention_type : {args.decoder_attention_type}")
    print(f"loss_name              : {args.loss_name}")
    print(f"augment                : {args.augment} ({args.aug_strength})")
    print(f"early_stopping_patience: {args.early_stopping_patience}")
    print(f"monitor                : {args.monitor}")
    print("=" * 100)

    train_loader, val_loader = build_loaders(args)

    model = SMPModel(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        classes=1,
        decoder_attention_type=args.decoder_attention_type,
    ).to(device)

    criterion = build_segmentation_loss(
        loss_name=args.loss_name,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        tversky_weight=args.tversky_weight,
        focal_weight=args.focal_weight,
        pos_weight=args.pos_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        focal_tversky_gamma=args.focal_tversky_gamma,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = build_scaler(use_amp)

    metrics_path = run_dir / "logs" / "metrics.csv"

    best_score = -1.0
    best_positive_dice = -1.0
    best_fp_constrained_score = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, args.threshold
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, use_amp, args.threshold
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
            "val_mean_dice_all": val_metrics["mean_dice_all"],
            "val_mean_iou_all": val_metrics["mean_iou_all"],
            "val_mean_dice_positive": val_metrics["mean_dice_positive"],
            "val_mean_iou_positive": val_metrics["mean_iou_positive"],
            "val_mean_dice_negative": val_metrics["mean_dice_negative"],
            "val_negative_fp_rate": val_metrics["negative_fp_rate"],
            "val_positive_image_recall": val_metrics["positive_image_recall"],
            "val_num_positive": val_metrics["num_positive"],
            "val_num_negative": val_metrics["num_negative"],
            "val_negative_false_positive": val_metrics["negative_false_positive"],
        }
        append_metrics(metrics_path, row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_pos_dice={val_metrics['mean_dice_positive']:.4f} "
            f"val_neg_fp={val_metrics['negative_fp_rate']:.4f} "
            f"val_pos_recall={val_metrics['positive_image_recall']:.4f}"
        )

        save_checkpoint(
            run_dir / "checkpoints" / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
            args,
        )

        current_score = float(row[args.monitor])
        if current_score > best_score + args.early_stopping_min_delta:
            best_score = current_score
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "checkpoints" / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
            )
            print(f"  -> Saved best_model.pth. {args.monitor}={best_score:.4f}")
        else:
            epochs_without_improvement += 1

        current_pos_dice = float(row["val_mean_dice_positive"])
        if current_pos_dice > best_positive_dice:
            best_positive_dice = current_pos_dice
            save_checkpoint(
                run_dir / "checkpoints" / "best_positive_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_positive_dice,
                args,
            )
            print(f"  -> Saved best_positive_model.pth. val_mean_dice_positive={best_positive_dice:.4f}")

        if float(row["val_negative_fp_rate"]) <= args.fp_constraint:
            fp_constrained_score = current_pos_dice
            if fp_constrained_score > best_fp_constrained_score:
                best_fp_constrained_score = fp_constrained_score
                save_checkpoint(
                    run_dir / "checkpoints" / "best_fp_constrained_model.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_fp_constrained_score,
                    args,
                )
                print(
                    "  -> Saved best_fp_constrained_model.pth. "
                    f"pos_dice={best_fp_constrained_score:.4f}, "
                    f"neg_fp={row['val_negative_fp_rate']:.4f}"
                )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"  -> Early stopping triggered. No improvement in {args.monitor} "
                f"for {epochs_without_improvement} epochs."
            )
            break

    print("=" * 100)
    print("[Done]")
    print(f"Best {args.monitor}          : {best_score:.4f}")
    print(f"Best positive dice      : {best_positive_dice:.4f}")
    print(f"Best FP-constrained pos : {best_fp_constrained_score:.4f}")
    print(f"Run dir                 : {run_dir}")
    print(f"Metrics                 : {metrics_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
