# scripts/visualize/check_siim_masks_by_area.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/siim_acr_pneumothorax"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/splits/siim_acr_pneumothorax"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="largest",
        choices=["largest", "smallest", "middle"],
        help="largest: 最大mask；smallest: 最小正样本mask；middle: 中等面积mask",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="Only visualize positive masks with mask_area >= min_area.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=3,
        help="Only dilate mask for visualization. It does not change training masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/siim_acr_mask_area_check"),
    )

    return parser.parse_args()


def read_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_image_path(image_dir: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{image_id}{ext}"
        if path.exists():
            return path

    candidates = [
        p for p in image_dir.glob(f"{image_id}.*")
        if p.suffix.lower() in IMAGE_EXTS
    ]

    if not candidates:
        raise FileNotFoundError(f"Image not found: {image_id}")

    return candidates[0]


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    return (mask > 127).astype(np.uint8)


def dilate_mask_for_display(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask

    img = Image.fromarray((mask * 255).astype(np.uint8))

    # MaxFilter size must be odd.
    size = radius * 2 + 1
    img = img.filter(ImageFilter.MaxFilter(size=size))

    return (np.asarray(img) > 127).astype(np.uint8)


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)

    red = np.zeros_like(overlay)
    red[..., 0] = 255.0

    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + red[mask_bool] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_figure(
        image_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        display_mask: np.ndarray,
        overlay: np.ndarray,
        mask_area: int,
        save_path: Path,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(display_mask, cmap="gray")
    ax2.set_title(f"Mask display | real area={mask_area}")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis("off")

    fig.suptitle(image_id, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    processed_root = args.processed_root
    image_dir = processed_root / "train"
    mask_dir = processed_root / "masks"
    summary_path = processed_root / "mask_summary.csv"
    split_path = args.split_root / f"{args.split}.txt"

    df = pd.read_csv(summary_path)
    df["ImageId"] = df["ImageId"].astype(str)

    split_ids = set(read_ids(split_path))

    df = df[df["ImageId"].isin(split_ids)].copy()
    df = df[df["has_mask"] == 1].copy()
    df = df[df["mask_area"] >= args.min_area].copy()

    if df.empty:
        raise RuntimeError("No positive samples found after filtering.")

    if args.mode == "largest":
        df = df.sort_values("mask_area", ascending=False)
    elif args.mode == "smallest":
        df = df.sort_values("mask_area", ascending=True)
    else:
        median_area = df["mask_area"].median()
        df["distance_to_median"] = (df["mask_area"] - median_area).abs()
        df = df.sort_values("distance_to_median", ascending=True)

    df = df.head(args.num)

    output_dir = args.output_dir / args.split / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[SIIM-ACR Mask Area Check]")
    print(f"split       : {args.split}")
    print(f"mode        : {args.mode}")
    print(f"num         : {args.num}")
    print(f"min_area    : {args.min_area}")
    print(f"dilate      : {args.dilate}")
    print(f"output_dir  : {output_dir}")
    print("-" * 80)
    print(df[["ImageId", "mask_area", "num_positive_rles"]].to_string(index=False))
    print("=" * 80)

    for _, row in df.iterrows():
        image_id = row["ImageId"]
        mask_area = int(row["mask_area"])

        image_path = find_image_path(image_dir, image_id)
        mask_path = mask_dir / f"{image_id}.png"

        image = load_image(image_path)
        mask = load_mask(mask_path)

        display_mask = dilate_mask_for_display(mask, args.dilate)
        overlay = make_overlay(image, display_mask)

        save_path = output_dir / f"{args.split}_{args.mode}_area{mask_area}_{image_id}.png"

        save_figure(
            image_id=image_id,
            image=image,
            mask=mask,
            display_mask=display_mask,
            overlay=overlay,
            mask_area=mask_area,
            save_path=save_path,
        )

    print("[Done]")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()