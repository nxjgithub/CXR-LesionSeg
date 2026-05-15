# scripts/preprocess/prepare_siim_acr.py
# -*- coding: utf-8 -*-

"""
Prepare SIIM-ACR Pneumothorax dataset.

Expected input structure:

CXR-LesionSeg/
├── data/
│   ├── processed/
│   │   └── siim_acr_pneumothorax/
│   │       ├── train/
│   │       ├── test/
│   │       ├── masks/
│   │       └── train-rle.csv
│   └── splits/
│       └── siim_acr_pneumothorax/

This script will:
1. Decode RLE annotations from train-rle.csv.
2. Merge multiple RLEs belonging to the same ImageId.
3. Generate one binary mask per training image.
4. Create train / val / test split files.
5. Save a mask_summary.csv for later checking.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SIIM-ACR Pneumothorax dataset masks and split files."
    )

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/siim_acr_pneumothorax"),
        help="Root directory of processed SIIM-ACR dataset.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/splits/siim_acr_pneumothorax"),
        help="Directory to save train/val/test split files.",
    )
    parser.add_argument(
        "--rle-csv",
        type=str,
        default="train-rle.csv",
        help="RLE annotation csv filename under processed-root.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="train",
        help="Training image directory name under processed-root.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="test",
        help="Test image directory name under processed-root.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="masks",
        help="Mask output directory name under processed-root.",
    )
    parser.add_argument(
        "--rle-height",
        type=int,
        default=1024,
        help=(
            "Height of the original RLE canvas. "
            "For SIIM-ACR, RLE is usually encoded on 1024x1024 images."
        ),
    )
    parser.add_argument(
        "--rle-width",
        type=int,
        default=1024,
        help=(
            "Width of the original RLE canvas. "
            "For SIIM-ACR, RLE is usually encoded on 1024x1024 images."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio split from training images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing mask files.",
    )

    return parser.parse_args()


def find_images(image_dir: Path) -> Dict[str, Path]:
    """
    Find image files and return {ImageId: image_path}.

    ImageId is assumed to be the filename stem.
    Example:
        1.2.276.xxx.png -> ImageId = 1.2.276.xxx
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images: Dict[str, Path] = {}
    duplicates: List[str] = []

    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        image_id = path.stem

        if image_id in images:
            duplicates.append(image_id)
            continue

        images[image_id] = path

    if duplicates:
        print(f"[Warning] Found duplicated image stems: {len(duplicates)}")
        print(f"[Warning] Example duplicated ImageId: {duplicates[:5]}")

    return images


def load_rle_csv(csv_path: Path) -> Dict[str, List[str]]:
    """
    Load train-rle.csv and group EncodedPixels by ImageId.

    Returns:
        {
            "ImageId_1": ["-1"],
            "ImageId_2": ["rle1", "rle2", ...],
        }
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"RLE csv does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lstrip("\ufeff") for col in df.columns]

    required_cols = {"ImageId", "EncodedPixels"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"RLE csv missing columns: {missing_cols}. "
            f"Current columns: {list(df.columns)}"
        )

    df["ImageId"] = df["ImageId"].astype(str).str.strip()
    df["EncodedPixels"] = df["EncodedPixels"].fillna("-1").astype(str).str.strip()

    rle_map = df.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
    return rle_map


def is_empty_rle(encoded_pixels: str) -> bool:
    """
    Check whether an RLE string means empty mask.
    """
    if encoded_pixels is None:
        return True

    value = str(encoded_pixels).strip().lower()
    return value in {"", "-1", "nan", "none"}


def rle_decode(
        encoded_pixels: str,
        rle_height: int,
        rle_width: int,
) -> np.ndarray:
    """
    Decode SIIM-ACR relative RLE into a binary mask.

    Important:
    SIIM-ACR Pneumothorax RLE is commonly stored as relative runs:
        gap_1 length_1 gap_2 length_2 ...

    It is not standard absolute-start RLE.

    The decoding process is:
        current_position += gap
        mask[current_position : current_position + length] = 1
        current_position += length

    The final flat mask is reshaped using column-major convention:
        flat.reshape((width, height)).T
    """
    if is_empty_rle(encoded_pixels):
        return np.zeros((rle_height, rle_width), dtype=np.uint8)

    numbers = encoded_pixels.split()
    if len(numbers) % 2 != 0:
        raise ValueError(
            f"Invalid RLE length: {len(numbers)}. RLE={encoded_pixels[:80]}..."
        )

    numbers = np.asarray(numbers, dtype=np.int64)

    gaps = numbers[0::2]
    lengths = numbers[1::2]

    total_pixels = rle_height * rle_width
    flat_mask = np.zeros(total_pixels, dtype=np.uint8)

    current_position = 0

    for gap, length in zip(gaps, lengths):
        current_position += int(gap)
        end_position = current_position + int(length)

        if current_position < 0 or end_position > total_pixels:
            raise ValueError(
                f"RLE exceeds canvas size. "
                f"current_position={current_position}, "
                f"end_position={end_position}, "
                f"canvas={rle_height}x{rle_width}={total_pixels}. "
                f"Try checking RLE mode or canvas size."
            )

        flat_mask[current_position:end_position] = 1
        current_position = end_position

    mask = flat_mask.reshape((rle_width, rle_height)).T
    return mask


def build_mask_from_rles(
        rles: List[str],
        rle_height: int,
        rle_width: int,
) -> Tuple[np.ndarray, int]:
    """
    Merge multiple RLEs of one image into one binary mask.

    Returns:
        mask: binary mask
        num_positive_rles: number of non-empty RLE segments
    """
    mask = np.zeros((rle_height, rle_width), dtype=np.uint8)

    positive_rles = [rle for rle in rles if not is_empty_rle(rle)]

    for rle in positive_rles:
        part_mask = rle_decode(rle, rle_height, rle_width)
        mask = np.logical_or(mask, part_mask).astype(np.uint8)

    return mask, len(positive_rles)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """
    Return image size as (height, width).
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return height, width


def resize_mask_if_needed(mask: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Resize mask to image size using nearest-neighbor interpolation.
    """
    if mask.shape == (target_height, target_width):
        return mask.astype(np.uint8)

    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))

    try:
        resample_method = Image.Resampling.NEAREST
    except AttributeError:
        resample_method = Image.NEAREST

    pil_mask = pil_mask.resize((target_width, target_height), resample=resample_method)
    resized = np.asarray(pil_mask)

    return (resized > 127).astype(np.uint8)


def save_mask(mask: np.ndarray, save_path: Path) -> None:
    """
    Save binary mask as png.

    Pixel values:
        background = 0
        lesion     = 255
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(save_path)


def stratified_train_val_split(
        image_ids: List[str],
        labels: Dict[str, int],
        val_ratio: float,
        seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Create a simple stratified train/val split by positive/negative label.

    labels:
        1 = image has pneumothorax mask
        0 = image has no pneumothorax mask
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio should be between 0 and 1, got {val_ratio}")

    rng = random.Random(seed)

    pos_ids = [image_id for image_id in image_ids if labels.get(image_id, 0) == 1]
    neg_ids = [image_id for image_id in image_ids if labels.get(image_id, 0) == 0]

    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    num_val_pos = int(round(len(pos_ids) * val_ratio))
    num_val_neg = int(round(len(neg_ids) * val_ratio))

    if len(pos_ids) > 0 and num_val_pos == 0:
        num_val_pos = 1

    if len(neg_ids) > 0 and num_val_neg == 0:
        num_val_neg = 1

    val_ids = set(pos_ids[:num_val_pos] + neg_ids[:num_val_neg])
    train_ids = [image_id for image_id in image_ids if image_id not in val_ids]
    val_ids = [image_id for image_id in image_ids if image_id in val_ids]

    return sorted(train_ids), sorted(val_ids)


def write_id_list(save_path: Path, image_ids: List[str]) -> None:
    """
    Write ImageId list to txt file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")


def main() -> None:
    args = parse_args()

    processed_root: Path = args.processed_root
    split_root: Path = args.split_root

    train_dir = processed_root / args.train_dir
    test_dir = processed_root / args.test_dir
    mask_dir = processed_root / args.mask_dir
    rle_csv_path = processed_root / args.rle_csv

    print("=" * 80)
    print("[SIIM-ACR Prepare]")
    print(f"processed_root : {processed_root}")
    print(f"train_dir      : {train_dir}")
    print(f"test_dir       : {test_dir}")
    print(f"mask_dir       : {mask_dir}")
    print(f"rle_csv        : {rle_csv_path}")
    print(f"split_root     : {split_root}")
    print(f"rle canvas     : {args.rle_height} x {args.rle_width}")
    print("=" * 80)

    train_images = find_images(train_dir)
    test_images = find_images(test_dir) if test_dir.exists() else {}

    print(f"[Info] Train images found: {len(train_images)}")
    print(f"[Info] Test images found : {len(test_images)}")

    rle_map = load_rle_csv(rle_csv_path)
    print(f"[Info] Unique ImageIds in RLE csv: {len(rle_map)}")

    train_image_ids = sorted(train_images.keys())
    csv_image_ids = set(rle_map.keys())

    missing_in_csv = sorted(set(train_image_ids) - csv_image_ids)
    missing_images = sorted(csv_image_ids - set(train_image_ids))

    if missing_in_csv:
        print(f"[Warning] Train images missing in csv: {len(missing_in_csv)}")
        print(f"[Warning] Example: {missing_in_csv[:5]}")

    if missing_images:
        print(f"[Warning] Csv ImageIds missing in train folder: {len(missing_images)}")
        print(f"[Warning] Example: {missing_images[:5]}")

    mask_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    generated_count = 0
    skipped_count = 0
    positive_count = 0
    negative_count = 0

    for idx, image_id in enumerate(train_image_ids, start=1):
        image_path = train_images[image_id]
        mask_path = mask_dir / f"{image_id}.png"

        if mask_path.exists() and not args.force:
            skipped_count += 1

            target_height, target_width = get_image_size(image_path)
            existing_mask = np.asarray(Image.open(mask_path))
            mask_area = int((existing_mask > 0).sum())
            has_mask = int(mask_area > 0)

            summary_rows.append(
                {
                    "ImageId": image_id,
                    "image_path": image_path.as_posix(),
                    "mask_path": mask_path.as_posix(),
                    "height": target_height,
                    "width": target_width,
                    "has_mask": has_mask,
                    "mask_area": mask_area,
                    "num_positive_rles": None,
                    "status": "skipped_existing",
                }
            )

            if has_mask:
                positive_count += 1
            else:
                negative_count += 1

            continue

        try:
            target_height, target_width = get_image_size(image_path)

            rles = rle_map.get(image_id, ["-1"])
            mask, num_positive_rles = build_mask_from_rles(
                rles=rles,
                rle_height=args.rle_height,
                rle_width=args.rle_width,
            )

            mask = resize_mask_if_needed(mask, target_height, target_width)
            save_mask(mask, mask_path)

            mask_area = int(mask.sum())
            has_mask = int(mask_area > 0)

            if has_mask:
                positive_count += 1
            else:
                negative_count += 1

            summary_rows.append(
                {
                    "ImageId": image_id,
                    "image_path": image_path.as_posix(),
                    "mask_path": mask_path.as_posix(),
                    "height": target_height,
                    "width": target_width,
                    "has_mask": has_mask,
                    "mask_area": mask_area,
                    "num_positive_rles": num_positive_rles,
                    "status": "generated",
                }
            )

            generated_count += 1

        except Exception as exc:
            raise RuntimeError(
                f"Failed to process ImageId={image_id}, image_path={image_path}"
            ) from exc

        if idx % 500 == 0 or idx == len(train_image_ids):
            print(f"[Progress] {idx}/{len(train_image_ids)} images processed")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = processed_root / "mask_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    labels = {
        row["ImageId"]: int(row["has_mask"])
        for row in summary_rows
    }

    train_ids, val_ids = stratified_train_val_split(
        image_ids=train_image_ids,
        labels=labels,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    test_ids = sorted(test_images.keys())

    write_id_list(split_root / "train.txt", train_ids)
    write_id_list(split_root / "val.txt", val_ids)
    write_id_list(split_root / "test.txt", test_ids)

    train_pos = sum(labels.get(image_id, 0) for image_id in train_ids)
    val_pos = sum(labels.get(image_id, 0) for image_id in val_ids)

    print("=" * 80)
    print("[Done]")
    print(f"Masks generated : {generated_count}")
    print(f"Masks skipped   : {skipped_count}")
    print(f"Positive images : {positive_count}")
    print(f"Negative images : {negative_count}")
    print(f"Summary csv     : {summary_path}")
    print("-" * 80)
    print(f"train.txt       : {split_root / 'train.txt'}")
    print(f"val.txt         : {split_root / 'val.txt'}")
    print(f"test.txt        : {split_root / 'test.txt'}")
    print("-" * 80)
    print(f"Train images    : {len(train_ids)} | positive: {train_pos}")
    print(f"Val images      : {len(val_ids)} | positive: {val_pos}")
    print(f"Test images     : {len(test_ids)}")
    print("=" * 80)


if __name__ == "__main__":
    main()