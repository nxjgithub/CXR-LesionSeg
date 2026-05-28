# scripts/evaluate/predict_siim_cls_gate.py
# -*- coding: utf-8 -*-
"""Predict image-level classification probabilities with a trained gate model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover
    smp = None
    _SMP_IMPORT_ERROR = exc
else:
    _SMP_IMPORT_ERROR = None

import sys
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "train"))
from train_siim_cls_gate import SMPEncoderClassifier, find_image_path, load_labels, load_split_ids


class PredictDataset(Dataset):
    def __init__(self, processed_root: Path, image_dir_name: str, image_ids: List[str], image_size: int, in_channels: int, labels: Optional[Dict[str, int]] = None) -> None:
        self.processed_root = processed_root
        self.image_dir = processed_root / image_dir_name
        self.image_ids = image_ids
        self.image_size = image_size
        self.in_channels = in_channels
        self.labels = labels or {}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_id = self.image_ids[idx]
        path = find_image_path(self.image_dir, image_id)
        img = Image.open(path).convert("L" if self.in_channels == 1 else "RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        arr = (arr - 0.5) / 0.5
        item: Dict[str, object] = {
            "image": torch.from_numpy(arr.astype(np.float32)),
            "image_id": image_id,
        }
        if image_id in self.labels:
            item["label"] = torch.tensor(float(self.labels[image_id]), dtype=torch.float32)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Predict SIIM-ACR classification gate probabilities.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--processed-root", type=str, default=None)
    parser.add_argument("--split-file", type=str, required=True)
    parser.add_argument("--image-dir-name", type=str, default=None, help="train for train/val folds, test for official test")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--mask-summary-csv", type=str, default=None, help="Optional, used only when labels are available.")
    parser.add_argument("--fold", type=int, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = SMPEncoderClassifier(
        cfg.get("encoder_name", "timm-efficientnet-b3"),
        cfg.get("encoder_weights", "imagenet"),
        int(cfg.get("in_channels", 3)),
        float(cfg.get("dropout", 0.2)),
    )
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, cfg = load_model(Path(args.checkpoint), device)

    processed_root = Path(args.processed_root or cfg.get("processed_root", "data/processed/siim_acr_pneumothorax"))
    image_dir_name = args.image_dir_name or cfg.get("image_dir_name", "train")
    image_size = int(cfg.get("image_size", 512))
    in_channels = int(cfg.get("in_channels", 3))
    image_ids = load_split_ids(Path(args.split_file))

    labels = None
    mask_csv = args.mask_summary_csv or cfg.get("mask_summary_csv")
    if mask_csv is not None and (processed_root / str(mask_csv)).exists():
        labels = load_labels(processed_root / str(mask_csv))

    ds = PredictDataset(processed_root, image_dir_name, image_ids, image_size, in_channels, labels=labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            logits_np = logits.cpu().numpy().tolist()
            labels_np = None
            if "label" in batch:
                labels_np = batch["label"].cpu().numpy().tolist()
            for idx, image_id in enumerate(batch["image_id"]):
                row = {
                    "image_id": image_id,
                    "cls_logit": float(logits_np[idx]),
                    "cls_prob": float(probs[idx]),
                }
                if args.fold is not None:
                    row["fold"] = int(args.fold)
                if labels_np is not None:
                    row["label"] = int(labels_np[idx])
                rows.append(row)

    fieldnames = []
    for key in ["fold", "image_id", "label", "cls_logit", "cls_prob"]:
        if rows and key in rows[0]:
            fieldnames.append(key)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote predictions: {args.output_csv} rows={len(rows)}")


if __name__ == "__main__":
    main()
