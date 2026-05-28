#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare SIIM-ACR 1024 PNG mirror dataset for CXR-LesionSeg.

Expected raw structure:
  data/raw/archive/train2_1/*.png        # official train images, SOPInstanceUID.png
  data/raw/archive/test2/*.png           # official Kaggle test images, ID_xxxxx.png
  data/raw/siim_acr_official/stage_2_train.csv
  data/raw/siim_acr_official/stage_2_sample_submission.csv

Output structure compatible with SIIMACRDataset:
  processed_root/train/*.png
  processed_root/test/*.png
  processed_root/masks/*.png
  processed_root/mask_summary.csv
  split_root/train.txt
  split_root/val.txt
  split_root/test.txt
  split_root/fold0/train.txt, val.txt, test.txt ...
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-root", default="data/raw/archive")
    p.add_argument("--train-dir", default=None, help="Default: <archive-root>/train2_1")
    p.add_argument("--test-dir", default=None, help="Default: <archive-root>/test2")
    p.add_argument("--train-csv", default="data/raw/siim_acr_official/stage_2_train.csv")
    p.add_argument("--sample-submission", default="data/raw/siim_acr_official/stage_2_sample_submission.csv")
    p.add_argument("--processed-root", default="data/processed/siim_acr_pneumothorax_1024png")
    p.add_argument("--split-root", default="data/splits/siim_acr_pneumothorax_1024png")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--link-mode",
        choices=["auto", "copy", "hardlink", "symlink"],
        default="auto",
        help="auto tries hardlink first then copy. hardlink saves disk space on the same volume.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_rle(value: object) -> str:
    if value is None:
        return "-1"
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return "-1"
    return s


def is_positive_rle(rle: str) -> bool:
    rle = normalize_rle(rle)
    return rle != "-1"


def rle_decode_siim(rle: str, height: int = 1024, width: int = 1024) -> np.ndarray:
    """Decode SIIM/Kaggle RLE. Order is top-to-bottom, then left-to-right, i.e. Fortran order."""
    rle = normalize_rle(rle)
    mask = np.zeros(height * width, dtype=np.uint8)
    if not is_positive_rle(rle):
        return mask.reshape((height, width), order="F")

    nums = [int(x) for x in rle.split()]
    if len(nums) % 2 != 0:
        raise ValueError(f"Invalid RLE length: {len(nums)} values")

    starts = np.asarray(nums[0::2], dtype=np.int64) - 1
    lengths = np.asarray(nums[1::2], dtype=np.int64)
    ends = starts + lengths

    for start, end in zip(starts, ends):
        if start < 0 or end > height * width:
            raise ValueError(f"RLE span out of bounds: start={start}, end={end}, size={height * width}")
        mask[start:end] = 1

    return mask.reshape((height, width), order="F")


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def safe_link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    src = src.resolve()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        os.link(src, dst)
        return

    if mode == "symlink":
        os.symlink(src, dst)
        return

    # auto: hardlink first, then copy. Hardlink usually works on Windows without admin if same disk.
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for image_id in ids:
            f.write(str(image_id).strip() + "\n")


def stratified_train_val_split(pos_ids: List[str], neg_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    pos = list(pos_ids)
    neg = list(neg_ids)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_val = int(round(len(pos) * val_ratio))
    n_neg_val = int(round(len(neg) * val_ratio))

    val_ids = pos[:n_pos_val] + neg[:n_neg_val]
    train_ids = pos[n_pos_val:] + neg[n_neg_val:]
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def make_stratified_folds(pos_ids: List[str], neg_ids: List[str], folds: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    rng = random.Random(seed)
    pos = list(pos_ids)
    neg = list(neg_ids)
    rng.shuffle(pos)
    rng.shuffle(neg)

    all_ids = set(pos + neg)
    out = []
    for fold in range(folds):
        val_ids = pos[fold::folds] + neg[fold::folds]
        val_set = set(val_ids)
        train_ids = [x for x in pos + neg if x not in val_set]
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        assert set(train_ids).isdisjoint(val_set)
        assert set(train_ids).union(val_set) == all_ids
        out.append((train_ids, val_ids))
    return out


def main() -> None:
    args = parse_args()

    archive_root = Path(args.archive_root)
    train_dir = Path(args.train_dir) if args.train_dir else archive_root / "train2_1"
    test_dir = Path(args.test_dir) if args.test_dir else archive_root / "test2"
    train_csv = Path(args.train_csv)
    sample_submission = Path(args.sample_submission)
    processed_root = Path(args.processed_root)
    split_root = Path(args.split_root)

    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    if not sample_submission.exists():
        raise FileNotFoundError(f"sample_submission not found: {sample_submission}")

    out_train_dir = processed_root / "train"
    out_test_dir = processed_root / "test"
    out_mask_dir = processed_root / "masks"
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_test_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    if not {"ImageId", "EncodedPixels"}.issubset(df.columns):
        raise KeyError(f"Expected columns ImageId and EncodedPixels in {train_csv}; got {list(df.columns)}")

    df["ImageId"] = df["ImageId"].astype(str)
    df["EncodedPixels"] = df["EncodedPixels"].map(normalize_rle)

    train_pngs = {p.stem: p for p in train_dir.glob("*.png")}
    test_pngs = {p.stem: p for p in test_dir.glob("*.png")}

    grouped = df.groupby("ImageId", sort=True)["EncodedPixels"].apply(list)
    csv_ids = list(grouped.index.astype(str))

    missing_train = [image_id for image_id in csv_ids if image_id not in train_pngs]
    extra_train = sorted(set(train_pngs) - set(csv_ids))

    print("=" * 80)
    print("[Input check]")
    print(f"train csv rows              : {len(df)}")
    print(f"unique train ImageId        : {len(csv_ids)}")
    print(f"train png files             : {len(train_pngs)}")
    print(f"missing train png by csv    : {len(missing_train)}")
    print(f"extra train png not in csv  : {len(extra_train)}")
    print(f"test png files              : {len(test_pngs)}")
    print("=" * 80)

    if missing_train:
        print("Missing examples:", missing_train[:20])
        raise RuntimeError("Some training images in CSV are missing from train png directory.")

    summary_rows = []
    pos_ids: List[str] = []
    neg_ids: List[str] = []

    for idx, image_id in enumerate(csv_ids, start=1):
        src_img = train_pngs[image_id]
        dst_img = out_train_dir / f"{image_id}.png"
        safe_link_or_copy(src_img, dst_img, args.link_mode, overwrite=args.overwrite)

        rles = [normalize_rle(x) for x in grouped.loc[image_id]]
        positive_rles = [x for x in rles if is_positive_rle(x)]

        mask = np.zeros((args.height, args.width), dtype=np.uint8)
        for rle in positive_rles:
            mask |= rle_decode_siim(rle, height=args.height, width=args.width)

        mask_path = out_mask_dir / f"{image_id}.png"
        if args.overwrite or not mask_path.exists():
            save_mask(mask, mask_path)

        mask_area = int(mask.sum())
        has_mask = int(mask_area > 0)
        if has_mask:
            pos_ids.append(image_id)
        else:
            neg_ids.append(image_id)

        summary_rows.append(
            {
                "ImageId": image_id,
                "image_path": str(dst_img),
                "mask_path": str(mask_path),
                "height": args.height,
                "width": args.width,
                "has_mask": has_mask,
                "mask_area": mask_area,
                "num_positive_rles": len(positive_rles),
                "status": "ok",
            }
        )

        if idx % 1000 == 0:
            print(f"processed train: {idx}/{len(csv_ids)}")

    summary_path = processed_root / "mask_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    sample = pd.read_csv(sample_submission)
    if "ImageId" not in sample.columns:
        raise KeyError(f"Expected ImageId column in sample submission: {sample_submission}")
    test_ids = sample["ImageId"].astype(str).tolist()
    missing_test = [image_id for image_id in test_ids if image_id not in test_pngs]
    if missing_test:
        print("Missing test examples:", missing_test[:20])
        raise RuntimeError(f"Missing {len(missing_test)} test images from {test_dir}")

    for idx, image_id in enumerate(test_ids, start=1):
        safe_link_or_copy(test_pngs[image_id], out_test_dir / f"{image_id}.png", args.link_mode, overwrite=args.overwrite)
        if idx % 1000 == 0:
            print(f"processed test: {idx}/{len(test_ids)}")

    train_ids, val_ids = stratified_train_val_split(pos_ids, neg_ids, args.val_ratio, args.seed)
    write_ids(split_root / "train.txt", train_ids)
    write_ids(split_root / "val.txt", val_ids)
    write_ids(split_root / "test.txt", test_ids)

    if args.folds > 1:
        folds = make_stratified_folds(pos_ids, neg_ids, args.folds, args.seed)
        for fold, (fold_train_ids, fold_val_ids) in enumerate(folds):
            fold_root = split_root / f"fold{fold}"
            write_ids(fold_root / "train.txt", fold_train_ids)
            write_ids(fold_root / "val.txt", fold_val_ids)
            write_ids(fold_root / "test.txt", test_ids)

    print("=" * 80)
    print("[Done]")
    print(f"processed_root : {processed_root}")
    print(f"split_root     : {split_root}")
    print(f"summary        : {summary_path}")
    print(f"positive train : {len(pos_ids)}")
    print(f"negative train : {len(neg_ids)}")
    print(f"main train     : {len(train_ids)}")
    print(f"main val       : {len(val_ids)}")
    print(f"test           : {len(test_ids)}")
    print(f"folds          : {args.folds}")
    print("=" * 80)


if __name__ == "__main__":
    main()
