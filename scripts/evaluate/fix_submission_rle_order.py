#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fix SIIM/Kaggle submission RLE order.

Typical use when a submitted CSV visually aligns only under row-major decode (order='C'):
  decode old RLE with order='C', then re-encode with Kaggle standard order='F'.

Input/Output columns:
  ImageId, EncodedPixels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

EMPTY_VALUES = {"", "-1", "nan", "NaN", "None", "none", "NULL", "null"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fix SIIM/Kaggle RLE order in a submission CSV")
    p.add_argument("--input-csv", required=True, help="Input submission csv")
    p.add_argument("--output-csv", required=True, help="Output fixed submission csv")
    p.add_argument("--height", type=int, default=1024, help="RLE canvas height")
    p.add_argument("--width", type=int, default=1024, help="RLE canvas width")
    p.add_argument(
        "--old-decode-order",
        choices=["C", "F"],
        default="C",
        help=(
            "How to decode the old/wrong CSV before re-encoding. "
            "Use C if your visualization showed the Alternative decode order=C panel is correct."
        ),
    )
    p.add_argument("--id-col", default="ImageId")
    p.add_argument("--rle-col", default="EncodedPixels")
    p.add_argument("--check-roundtrip", action="store_true")
    return p.parse_args()


def is_empty_rle(x) -> bool:
    return str(x).strip() in EMPTY_VALUES


def decode_rle(rle: str, shape: Tuple[int, int], order: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if is_empty_rle(rle):
        return mask.reshape((h, w), order=order)

    nums = [int(v) for v in str(rle).strip().split()]
    if len(nums) % 2 != 0:
        raise ValueError(f"Invalid RLE length={len(nums)}")

    starts = np.asarray(nums[0::2], dtype=np.int64) - 1
    lengths = np.asarray(nums[1::2], dtype=np.int64)
    for start, length in zip(starts, lengths):
        if start < 0:
            raise ValueError(f"Invalid RLE start={start + 1}")
        end = start + length
        if end > h * w:
            raise ValueError(f"RLE run exceeds mask size: end={end}, size={h*w}")
        mask[start:end] = 1
    return mask.reshape((h, w), order=order)


def encode_rle_kaggle(mask: np.ndarray) -> str:
    """Kaggle/SIIM standard: column-major order, i.e. mask.reshape(-1, order='F')."""
    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return "-1"

    pixels = mask.reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(int(x)) for x in changes)


def rle_area(rle: str) -> int:
    if is_empty_rle(rle):
        return 0
    nums = [int(v) for v in str(rle).strip().split()]
    return int(sum(nums[1::2]))


def main() -> None:
    args = parse_args()
    inp = Path(args.input_csv)
    outp = Path(args.output_csv)

    df = pd.read_csv(inp)
    required = {args.id_col, args.rle_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}; columns={df.columns.tolist()}")

    fixed = []
    area_changed = 0
    n_nonempty = 0
    for rle in df[args.rle_col].astype(str).tolist():
        old_area = rle_area(rle)
        mask = decode_rle(rle, (args.height, args.width), order=args.old_decode_order)
        new_rle = encode_rle_kaggle(mask)
        new_area = rle_area(new_rle)
        if old_area != new_area:
            area_changed += 1
        if new_area > 0:
            n_nonempty += 1
        if args.check_roundtrip:
            rt = decode_rle(new_rle, (args.height, args.width), order="F")
            if not np.array_equal(mask, rt):
                raise RuntimeError("Roundtrip check failed")
        fixed.append(new_rle)

    df[args.rle_col] = fixed
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    print("[Done]", outp)
    print("rows:", len(df))
    print("nonempty:", n_nonempty)
    print("area_changed:", area_changed, "(should be 0 for pure order fix)")
    print("old_decode_order:", args.old_decode_order)
    print("new_encode_order: F / Kaggle standard")


if __name__ == "__main__":
    main()
