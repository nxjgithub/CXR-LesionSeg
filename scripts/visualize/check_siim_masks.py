# scripts/visualize/check_siim_masks.py
# -*- coding: utf-8 -*-

"""
Visual check for SIIM-ACR masks.

This script samples positive / negative cases from train or val split,
then saves image-mask overlay figures for manual inspection.

Expected project structure:

CXR-LesionSeg/
├── data/
│   ├── processed/
│   │   └── siim_acr_pneumothorax/
│   │       ├── train/
│   │       ├── masks/
│   │       └── mask_summary.csv
│   └── splits/
│       └── siim_acr_pneumothorax/
│           ├── train.txt
│           └── val.txt
└── outputs/
    └── visualizations/
        └── siim_acr_mask_check/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SIIM-ACR images and generated masks."
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/siim_acr_pneumothorax"),
        help="Processed SIIM-ACR dataset root.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/splits/siim_acr_pneumothorax"),
        help="Split directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to sample from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/siim_acr_mask_check"),
        help="Directory to save visualization results.",
    )
    parser.add_argument(
        "--num-pos",
        type=int,
        default=8,
        help="Number of positive samples to visualize.",
    )
    parser.add_argument(
        "--num-neg",
        type=int,
        default=4,
        help="Number of negative samples to visualize.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Mask overlay transparency.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def read_id_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    return ids


def find_image_path(image_dir: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{image_id}{ext}"
        if path.exists():
            return path

    candidates = list(image_dir.glob(f"{image_id}.*"))
    candidates = [p for p in candidates if p.suffix.lower() in IMAGE_EXTS]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Image not found for ImageId={image_id} under {image_dir}")


def load_image_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def load_mask_binary(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Mask does not exist: {path}")

    mask = Image.open(path).convert("L")
    mask = np.asarray(mask)

    return (mask > 127).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Overlay binary mask on image.

    Mask area is shown in red.
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask shape mismatch: image={image.shape}, mask={mask.shape}"
        )

    overlay = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)

    red = np.zeros_like(overlay)
    red[..., 0] = 255.0

    overlay[mask_bool] = (
            image.astype(np.float32)[mask_bool] * (1.0 - alpha)
            + red[mask_bool] * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_single_visualization(
        image_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        overlay: np.ndarray,
        save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"Mask | area={int(mask.sum())}")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis("off")

    fig.suptitle(image_id, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_contact_sheet(
        rows: List[Tuple[str, np.ndarray, np.ndarray]],
        save_path: Path,
) -> None:
    """
    Save a compact overview figure.

    rows:
        [(image_id, image_rgb, overlay_rgb), ...]
    """
    if not rows:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_rows = len(rows)
    fig = plt.figure(figsize=(8, max(3, num_rows * 2.2)))

    for i, (image_id, image, overlay) in enumerate(rows):
        ax1 = fig.add_subplot(num_rows, 2, i * 2 + 1)
        ax1.imshow(image, cmap="gray")
        ax1.set_title(f"{image_id} | image", fontsize=8)
        ax1.axis("off")

        ax2 = fig.add_subplot(num_rows, 2, i * 2 + 2)
        ax2.imshow(overlay)
        ax2.set_title("overlay", fontsize=8)
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    processed_root = args.processed_root
    split_root = args.split_root
    output_dir = args.output_dir

    image_dir = processed_root / "train"
    mask_dir = processed_root / "masks"
    summary_path = processed_root / "mask_summary.csv"
    split_path = split_root / f"{args.split}.txt"

    print("=" * 80)
    print("[SIIM-ACR Mask Check]")
    print(f"processed_root : {processed_root}")
    print(f"image_dir      : {image_dir}")
    print(f"mask_dir       : {mask_dir}")
    print(f"summary_csv    : {summary_path}")
    print(f"split_file     : {split_path}")
    print(f"output_dir     : {output_dir}")
    print("=" * 80)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"mask_summary.csv not found: {summary_path}. "
            "Please run prepare_siim_acr.py first."
        )

    df = pd.read_csv(summary_path)
    df["ImageId"] = df["ImageId"].astype(str)

    split_ids = read_id_list(split_path)
    split_set = set(split_ids)

    df_split = df[df["ImageId"].isin(split_set)].copy()

    if df_split.empty:
        raise RuntimeError(f"No samples found for split={args.split}")

    pos_ids = df_split[df_split["has_mask"] == 1]["ImageId"].tolist()
    neg_ids = df_split[df_split["has_mask"] == 0]["ImageId"].tolist()

    rng = random.Random(args.seed)
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    sampled_pos = pos_ids[: min(args.num_pos, len(pos_ids))]
    sampled_neg = neg_ids[: min(args.num_neg, len(neg_ids))]

    sampled_ids = sampled_pos + sampled_neg

    print(f"[Info] Split images     : {len(df_split)}")
    print(f"[Info] Positive images  : {len(pos_ids)}")
    print(f"[Info] Negative images  : {len(neg_ids)}")
    print(f"[Info] Sampled positive : {len(sampled_pos)}")
    print(f"[Info] Sampled negative : {len(sampled_neg)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    contact_rows: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for image_id in sampled_ids:
        image_path = find_image_path(image_dir, image_id)
        mask_path = mask_dir / f"{image_id}.png"

        image = load_image_rgb(image_path)
        mask = load_mask_binary(mask_path)
        overlay = make_overlay(image, mask, alpha=args.alpha)

        label_name = "positive" if mask.sum() > 0 else "negative"
        save_path = output_dir / f"{args.split}_{label_name}_{image_id}.png"

        save_single_visualization(
            image_id=image_id,
            image=image,
            mask=mask,
            overlay=overlay,
            save_path=save_path,
        )

        contact_rows.append((image_id, image, overlay))

    contact_sheet_path = output_dir / f"{args.split}_contact_sheet.png"
    save_contact_sheet(contact_rows, contact_sheet_path)

    print("=" * 80)
    print("[Done]")
    print(f"Saved individual figures to: {output_dir}")
    print(f"Saved contact sheet to     : {contact_sheet_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()