# scripts/train/train_siim_cls_gate.py
# -*- coding: utf-8 -*-
"""Train an independent image-level classification gate for SIIM-ACR.

This is intentionally independent from the segmentation decoder. The goal is to
learn whether an image contains pneumothorax, then gate segmentation RLE output.

Recommended use:
  one model per fold -> validation predictions are OOF classifier probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover
    smp = None
    _SMP_IMPORT_ERROR = exc
else:
    _SMP_IMPORT_ERROR = None

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


@dataclass
class Config:
    processed_root: str
    train_split_file: str
    val_split_file: str
    output_dir: str
    encoder_name: str = "timm-efficientnet-b3"
    encoder_weights: Optional[str] = "imagenet"
    image_size: int = 512
    in_channels: int = 3
    image_dir_name: str = "train"
    mask_summary_csv: str = "mask_summary.csv"
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 8
    lr: float = 2e-4
    weight_decay: float = 3e-4
    dropout: float = 0.2
    pos_weight: Optional[float] = None
    balanced_sampling: bool = True
    positive_sample_ratio: float = 0.5
    augment: bool = True
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    early_stopping_patience: int = 8
    metric_threshold: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train SIIM-ACR image-level classification gate.")
    parser.add_argument("--processed-root", type=str, default="data/processed/siim_acr_pneumothorax")
    parser.add_argument("--train-split-file", type=str, required=True)
    parser.add_argument("--val-split-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--encoder-name", type=str, default="timm-efficientnet-b3")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--image-dir-name", type=str, default="train")
    parser.add_argument("--mask-summary-csv", type=str, default="mask_summary.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--positive-sample-ratio", type=float, default=0.5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"Empty split file: {path}")
    return ids


def load_labels(mask_summary_path: Path) -> Dict[str, int]:
    if not mask_summary_path.exists():
        raise FileNotFoundError(f"mask_summary.csv not found: {mask_summary_path}")
    labels: Dict[str, int] = {}
    with mask_summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {mask_summary_path}")
        cols = {c.strip().lstrip("\ufeff") for c in reader.fieldnames}
        id_col = "ImageId" if "ImageId" in cols else "image_id"
        if id_col not in cols:
            raise ValueError(f"Missing ImageId/image_id column in {mask_summary_path}")
        if "has_mask" not in cols:
            raise ValueError(f"Missing has_mask column in {mask_summary_path}")
        for row in reader:
            image_id = str(row[id_col]).strip()
            raw = str(row["has_mask"]).strip().lower()
            labels[image_id] = 1 if raw in {"1", "true", "yes", "y"} else 0
    return labels


def find_image_path(image_dir: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    candidates = [p for p in image_dir.glob(f"{image_id}.*") if p.suffix.lower() in IMAGE_EXTS]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Image not found: {image_id} under {image_dir}")


def random_image_augment(img: Image.Image) -> Image.Image:
    # Light augmentations suitable for CXR. Keep geometry conservative.
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.35:
        img = img.rotate(random.uniform(-5.0, 5.0), resample=Image.BILINEAR, fillcolor=0)
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))
    return img


class SIIMClassificationDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        split_file: Path,
        image_dir_name: str,
        image_size: int,
        in_channels: int,
        labels: Dict[str, int],
        augment: bool = False,
    ) -> None:
        self.processed_root = processed_root
        self.image_dir = processed_root / image_dir_name
        self.image_ids = load_split_ids(split_file)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.labels = labels
        self.augment = augment

        missing = [x for x in self.image_ids if x not in self.labels]
        if missing:
            raise ValueError(f"{len(missing)} split IDs missing labels. Example: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_id = self.image_ids[index]
        path = find_image_path(self.image_dir, image_id)
        img = Image.open(path).convert("L" if self.in_channels == 1 else "RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if self.augment:
            img = random_image_augment(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        # Match the existing SIIM tensor convention: roughly [-1, 1].
        arr = (arr - 0.5) / 0.5
        image = torch.from_numpy(arr.astype(np.float32))
        label = torch.tensor(float(self.labels[image_id]), dtype=torch.float32)
        return {"image": image, "label": label, "image_id": image_id}


class SMPEncoderClassifier(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: Optional[str], in_channels: int, dropout: float) -> None:
        super().__init__()
        if smp is None:
            raise RuntimeError(f"segmentation_models_pytorch import failed: {_SMP_IMPORT_ERROR}")
        weights = None if encoder_weights in {None, "", "none", "None"} else encoder_weights
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=int(in_channels),
            depth=5,
            weights=weights,
        )
        out_ch = int(self.encoder.out_channels[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = features[-1]
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x).squeeze(1)


def make_sampler(dataset: SIIMClassificationDataset, positive_sample_ratio: float) -> WeightedRandomSampler:
    labels = [dataset.labels[i] for i in dataset.image_ids]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both positive and negative samples. pos={n_pos}, neg={n_neg}")
    pos_w = float(positive_sample_ratio) / n_pos
    neg_w = float(1.0 - positive_sample_ratio) / n_neg
    weights = [pos_w if y == 1 else neg_w for y in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int32)
    labels_i = labels.astype(np.int32)
    tp = int(((preds == 1) & (labels_i == 1)).sum())
    fp = int(((preds == 1) & (labels_i == 0)).sum())
    tn = int(((preds == 0) & (labels_i == 0)).sum())
    fn = int(((preds == 0) & (labels_i == 1)).sum())
    eps = 1e-12
    metrics: Dict[str, float] = {
        "acc": (tp + tn) / max(1, len(labels_i)),
        "precision": tp / max(eps, tp + fp),
        "recall": tp / max(eps, tp + fn),
        "specificity": tn / max(eps, tn + fp),
        "negative_fp_rate": fp / max(eps, tn + fp),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    if roc_auc_score is not None and len(np.unique(labels_i)) == 2:
        metrics["auc"] = float(roc_auc_score(labels_i, probs))
    else:
        metrics["auc"] = float("nan")
    if average_precision_score is not None and len(np.unique(labels_i)) == 2:
        metrics["ap"] = float(average_precision_score(labels_i, probs))
    else:
        metrics["ap"] = float("nan")
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp: bool,
    threshold: float,
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    all_labels: List[float] = []
    all_probs: List[float] = []
    pred_rows: List[Dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            assert optimizer is not None
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = int(images.shape[0])
        total_loss += float(loss.detach().item()) * bs
        total_n += bs
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())
        for image_id, label, logit, prob in zip(
            batch["image_id"], labels_np.tolist(), logits.detach().cpu().numpy().tolist(), probs.tolist()
        ):
            pred_rows.append({
                "image_id": image_id,
                "label": int(label),
                "cls_logit": float(logit),
                "cls_prob": float(prob),
            })

    labels_np = np.asarray(all_labels, dtype=np.float32)
    probs_np = np.asarray(all_probs, dtype=np.float32)
    metrics = binary_metrics(labels_np, probs_np, threshold=threshold)
    return total_loss / max(1, total_n), metrics, pred_rows


def save_checkpoint(path: Path, model: nn.Module, config: Config, epoch: int, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def write_predictions(path: Path, rows: List[Dict[str, object]], fold: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "label", "cls_logit", "cls_prob"]
    if fold is not None:
        fieldnames = ["fold"] + fieldnames
        rows = [{"fold": fold, **row} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = Config(
        processed_root=args.processed_root,
        train_split_file=args.train_split_file,
        val_split_file=args.val_split_file,
        output_dir=args.output_dir,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        image_size=args.image_size,
        in_channels=args.in_channels,
        image_dir_name=args.image_dir_name,
        mask_summary_csv=args.mask_summary_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pos_weight=args.pos_weight,
        balanced_sampling=args.balanced_sampling,
        positive_sample_ratio=args.positive_sample_ratio,
        augment=not args.no_augment,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        early_stopping_patience=args.early_stopping_patience,
        metric_threshold=args.metric_threshold,
    )
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    processed_root = Path(cfg.processed_root)
    labels = load_labels(processed_root / cfg.mask_summary_csv)
    train_ds = SIIMClassificationDataset(
        processed_root, Path(cfg.train_split_file), cfg.image_dir_name, cfg.image_size,
        cfg.in_channels, labels, augment=cfg.augment,
    )
    val_ds = SIIMClassificationDataset(
        processed_root, Path(cfg.val_split_file), cfg.image_dir_name, cfg.image_size,
        cfg.in_channels, labels, augment=False,
    )

    train_labels = [labels[x] for x in train_ds.image_ids]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight_value = float(cfg.pos_weight) if cfg.pos_weight is not None else (n_neg / max(1, n_pos))
    print(f"[Data] train={len(train_ds)} pos={n_pos} neg={n_neg} pos_weight={pos_weight_value:.4f}")
    print(f"[Data] val={len(val_ds)} pos={sum(labels[x] for x in val_ds.image_ids)} neg={len(val_ds)-sum(labels[x] for x in val_ds.image_ids)}")

    sampler = make_sampler(train_ds, cfg.positive_sample_ratio) if cfg.balanced_sampling else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = SMPEncoderClassifier(cfg.encoder_name, cfg.encoder_weights, cfg.in_channels, cfg.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    metrics_path = output_dir / "metrics.csv"
    metrics_rows: List[Dict[str, object]] = []
    best_score = -math.inf
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics, _ = run_epoch(
            model, train_loader, device, criterion, optimizer, scaler, cfg.amp, cfg.metric_threshold
        )
        scheduler.step()
        val_loss, val_metrics, val_pred_rows = run_epoch(
            model, val_loader, device, criterion, None, None, cfg.amp, cfg.metric_threshold
        )

        # AUC is the right selection metric for a gate; fallback to -val_loss if AUC is unavailable.
        score = val_metrics.get("auc", float("nan"))
        if not math.isfinite(score):
            score = -val_loss

        row: Dict[str, object] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auc": train_metrics.get("auc"),
            "val_auc": val_metrics.get("auc"),
            "train_ap": train_metrics.get("ap"),
            "val_ap": val_metrics.get("ap"),
            "train_acc": train_metrics["acc"],
            "val_acc": val_metrics["acc"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_specificity": train_metrics["specificity"],
            "val_specificity": val_metrics["specificity"],
            "val_negative_fp_rate": val_metrics["negative_fp_rate"],
            "seconds": time.time() - t0,
        }
        metrics_rows.append(row)

        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics.get('auc', float('nan')):.4f} "
            f"val_ap={val_metrics.get('ap', float('nan')):.4f} "
            f"val_recall@{cfg.metric_threshold:.2f}={val_metrics['recall']:.4f} "
            f"val_spec@{cfg.metric_threshold:.2f}={val_metrics['specificity']:.4f}"
        )

        save_checkpoint(output_dir / "checkpoints" / "last_model.pth", model, cfg, epoch, val_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(output_dir / "checkpoints" / "best_model.pth", model, cfg, epoch, val_metrics)
            write_predictions(output_dir / "val_predictions_best.csv", val_pred_rows)
            print(f"  -> Saved best model. best_score={best_score:.6f}")
        else:
            bad_epochs += 1

        if cfg.early_stopping_patience > 0 and bad_epochs >= cfg.early_stopping_patience:
            print(f"[EarlyStop] No improvement for {bad_epochs} epochs. best_epoch={best_epoch}")
            break

    print(f"[Done] Best epoch: {best_epoch}, best_score={best_score:.6f}")


if __name__ == "__main__":
    main()
