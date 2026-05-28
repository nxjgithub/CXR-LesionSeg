# -*- coding: utf-8 -*-
"""
Visualize Kaggle SIIM-ACR submission RLE masks on official test images.

Purpose:
  1) Preview top-k non-empty predictions from a submission.csv.
  2) Check whether Kaggle RLE decoding direction is correct.
  3) Compare standard Kaggle column-major decode (order='F') with row-major decode (order='C').

Expected submission format:
  ImageId,EncodedPixels

Typical SIIM-ACR submission mask size is 1024 x 1024. If your processed test PNGs are 512 x 512,
the script decodes RLE at 1024 x 1024 and resizes the mask to the image size for overlay.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
EMPTY_VALUES = {"", "-1", "nan", "NaN", "None", "none", "NULL", "null"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualize Kaggle SIIM submission RLE predictions")
    p.add_argument("--submission", required=True, help="Path to Kaggle submission csv: ImageId,EncodedPixels")
    p.add_argument("--image-root", required=True, help="Directory containing official test PNG/JPG images")
    p.add_argument("--output-dir", required=True, help="Directory to save visualizations")

    p.add_argument("--cls-probs-csv", default=None, help="Optional csv with ImageId/image_id and cls_prob")
    p.add_argument("--sort-by", default="submission_order",
                   choices=["submission_order", "cls_prob", "area"],
                   help="How to rank non-empty predictions before taking top-k")
    p.add_argument("--top-k", type=int, default=50, help="Number of non-empty predictions to visualize")
    p.add_argument("--image-ids", nargs="*", default=None, help="Optional explicit ImageIds to visualize")

    p.add_argument("--rle-height", type=int, default=1024, help="RLE mask height used in submission encoding")
    p.add_argument("--rle-width", type=int, default=1024, help="RLE mask width used in submission encoding")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha, 0~1")
    p.add_argument("--make-contact-sheet", action="store_true", help="Also save a contact sheet")
    p.add_argument("--contact-cols", type=int, default=2, help="Contact sheet columns")
    return p.parse_args()


def is_empty_rle(x) -> bool:
    return str(x).strip() in EMPTY_VALUES


def rle_area(rle: str) -> int:
    if is_empty_rle(rle):
        return 0
    nums = [int(v) for v in str(rle).strip().split()]
    if len(nums) % 2 != 0:
        raise ValueError(f"Invalid RLE length: {len(nums)} values")
    return int(sum(nums[1::2]))


def decode_rle(rle: str, shape: Tuple[int, int], order: str = "F") -> np.ndarray:
    """Decode SIIM/Kaggle RLE. Standard Kaggle mask uses column-major order='F'."""
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if is_empty_rle(rle):
        return mask.reshape((h, w), order=order)

    nums = [int(v) for v in str(rle).strip().split()]
    if len(nums) % 2 != 0:
        raise ValueError(f"Invalid RLE: odd number of integers: {len(nums)}")

    starts = np.asarray(nums[0::2], dtype=np.int64) - 1
    lengths = np.asarray(nums[1::2], dtype=np.int64)

    for start, length in zip(starts, lengths):
        if start < 0:
            raise ValueError(f"Invalid RLE start < 1: {start + 1}")
        end = start + length
        if end > h * w:
            raise ValueError(f"RLE run exceeds mask size: end={end}, size={h*w}")
        mask[start:end] = 1

    return mask.reshape((h, w), order=order)


def find_image(image_root: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTS:
        p = image_root / f"{image_id}{ext}"
        if p.exists():
            return p
    matches = []
    for ext in IMAGE_EXTS:
        matches.extend(image_root.glob(f"**/{image_id}{ext}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found: {image_id} under {image_root}")


def load_font(size: int = 18):
    # Default PIL font is always available. Use it to avoid environment font issues.
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def add_title(img: Image.Image, title: str, height: int = 34) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), (255, 255, 255))
    out.paste(img.convert("RGB"), (0, height))
    d = ImageDraw.Draw(out)
    d.text((8, 9), title, fill=(0, 0, 0), font=load_font())
    return out


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    if mask.shape != (base.height, base.width):
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((base.width, base.height), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    else:
        mask = mask > 0

    arr = np.asarray(base).astype(np.float32)
    # Red overlay without requiring matplotlib/cv2.
    red = np.zeros_like(arr)
    red[..., 0] = 255.0
    m = mask[..., None]
    arr = np.where(m, arr * (1.0 - alpha) + red * alpha, arr)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_panel(image: Image.Image, mask_f: np.ndarray, mask_c: np.ndarray, title: str, area: int, cls_prob: Optional[float]) -> Image.Image:
    max_w = 360
    def resize_for_panel(im: Image.Image) -> Image.Image:
        scale = max_w / im.width
        return im.resize((max_w, int(im.height * scale)), resample=Image.Resampling.BILINEAR)

    original = resize_for_panel(image.convert("RGB"))
    overlay_f = resize_for_panel(overlay_mask(image, mask_f))
    overlay_c = resize_for_panel(overlay_mask(image, mask_c))

    prob_txt = "NA" if cls_prob is None or (isinstance(cls_prob, float) and math.isnan(cls_prob)) else f"{cls_prob:.6f}"
    original = add_title(original, f"Original | {title}")
    overlay_f = add_title(overlay_f, f"Kaggle decode order=F | area={area} | cls={prob_txt}")
    overlay_c = add_title(overlay_c, "Alternative decode order=C")

    gap = 8
    h = max(original.height, overlay_f.height, overlay_c.height)
    w = original.width + overlay_f.width + overlay_c.width + gap * 2
    out = Image.new("RGB", (w, h), (245, 245, 245))
    x = 0
    out.paste(original, (x, 0)); x += original.width + gap
    out.paste(overlay_f, (x, 0)); x += overlay_f.width + gap
    out.paste(overlay_c, (x, 0))
    return out


def make_contact_sheet(paths: List[Path], output_path: Path, cols: int = 2) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if not imgs:
        return
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * max_w, rows * max_h), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * max_w, r * max_h))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(submission_path)
    required = {"ImageId", "EncodedPixels"}
    missing = required - set(sub.columns)
    if missing:
        raise ValueError(f"Submission missing columns: {missing}. Columns={sub.columns.tolist()}")

    sub = sub.copy()
    sub["_submission_order"] = np.arange(len(sub))
    sub["area"] = sub["EncodedPixels"].apply(rle_area)
    sub["nonempty"] = sub["area"] > 0

    if args.cls_probs_csv:
        cls = pd.read_csv(args.cls_probs_csv)
        if "image_id" in cls.columns and "ImageId" not in cls.columns:
            cls = cls.rename(columns={"image_id": "ImageId"})
        if "cls_prob" not in cls.columns:
            raise ValueError(f"cls_prob not found in cls csv. Columns={cls.columns.tolist()}")
        sub = sub.merge(cls[["ImageId", "cls_prob"]], on="ImageId", how="left")
    else:
        sub["cls_prob"] = np.nan

    if args.image_ids:
        pick = sub[sub["ImageId"].isin(args.image_ids)].copy()
        missing_ids = sorted(set(args.image_ids) - set(pick["ImageId"]))
        if missing_ids:
            print(f"[Warning] ImageIds not found in submission: {missing_ids}")
    else:
        pick = sub[sub["nonempty"]].copy()
        if args.sort_by == "cls_prob":
            pick = pick.sort_values(["cls_prob", "area"], ascending=[False, False])
        elif args.sort_by == "area":
            pick = pick.sort_values(["area", "cls_prob"], ascending=[False, False])
        else:
            pick = pick.sort_values("_submission_order")
        pick = pick.head(args.top_k)

    print("=" * 80)
    print(f"submission       : {submission_path}")
    print(f"image_root       : {image_root}")
    print(f"output_dir       : {output_dir}")
    print(f"rows             : {len(sub)}")
    print(f"nonempty total   : {int(sub['nonempty'].sum())}")
    print(f"visualize count  : {len(pick)}")
    print(f"rle shape        : {args.rle_height} x {args.rle_width}")
    print("=" * 80)

    saved_paths: List[Path] = []
    for rank, row in enumerate(pick.itertuples(index=False), start=1):
        image_id = str(row.ImageId)
        rle = str(row.EncodedPixels)
        area = int(row.area)
        cls_prob = float(row.cls_prob) if pd.notna(row.cls_prob) else np.nan

        image_path = find_image(image_root, image_id)
        image = Image.open(image_path).convert("RGB")
        mask_f = decode_rle(rle, (args.rle_height, args.rle_width), order="F")
        mask_c = decode_rle(rle, (args.rle_height, args.rle_width), order="C")

        title = f"rank={rank:03d} | {image_id} | img={image.width}x{image.height}"
        panel = make_panel(image, mask_f, mask_c, title=title, area=area, cls_prob=cls_prob)
        out_path = output_dir / f"rank_{rank:03d}_{image_id}_area_{area}_cls_{cls_prob:.6f}.png"
        panel.save(out_path)
        saved_paths.append(out_path)
        print(f"[{rank:03d}] {image_id} area={area} cls={cls_prob:.6f} image={image_path} -> {out_path}")

    if args.make_contact_sheet:
        sheet_path = output_dir / "contact_sheet.png"
        make_contact_sheet(saved_paths, sheet_path, cols=args.contact_cols)
        print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
