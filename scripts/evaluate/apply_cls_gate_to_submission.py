# scripts/evaluate/apply_cls_gate_to_submission.py
# -*- coding: utf-8 -*-
"""Apply a classification gate to an existing Kaggle submission CSV.

Rows with cls_prob < gate_threshold are converted to empty mask (-1).
Existing empty masks are kept empty.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Apply classification gate to Kaggle submission.")
    parser.add_argument("--input-submission", type=str, required=True)
    parser.add_argument("--cls-probs-csv", type=str, required=True)
    parser.add_argument("--output-submission", type=str, required=True)
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--keep-top-k", type=int, default=None, help="Alternative to threshold: keep only top K classifier-positive rows.")
    parser.add_argument("--id-column", type=str, default="ImageId")
    parser.add_argument("--rle-column", type=str, default="EncodedPixels")
    parser.add_argument("--prob-column", type=str, default="cls_prob")
    return parser.parse_args()


def empty_rle(x: str) -> bool:
    x = str(x).strip()
    return x in {"", "-1", "nan", "NaN", "None"}


def main() -> None:
    args = parse_args()
    if args.gate_threshold is None and args.keep_top_k is None:
        raise ValueError("Specify either --gate-threshold or --keep-top-k")
    if args.gate_threshold is not None and args.keep_top_k is not None:
        raise ValueError("Use only one of --gate-threshold or --keep-top-k")

    probs: Dict[str, float] = {}
    with Path(args.cls_probs_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id") or row.get("ImageId")
            probs[image_id] = float(row[args.prob_column])

    with Path(args.input_submission).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = reader.fieldnames or [args.id_column, args.rle_column]

    if args.keep_top_k is not None:
        ranked = sorted(((probs.get(row[args.id_column], -1.0), row[args.id_column]) for row in rows), reverse=True)
        keep_ids = {image_id for _, image_id in ranked[:args.keep_top_k]}
    else:
        keep_ids = {image_id for image_id, p in probs.items() if p >= float(args.gate_threshold)}

    missing = 0
    originally_nonempty = 0
    final_nonempty = 0
    gated_nonempty_to_empty = 0
    for row in rows:
        image_id = row[args.id_column]
        rle = row[args.rle_column]
        if image_id not in probs:
            missing += 1
        was_nonempty = not empty_rle(rle)
        if was_nonempty:
            originally_nonempty += 1
        if image_id not in keep_ids:
            if was_nonempty:
                gated_nonempty_to_empty += 1
            row[args.rle_column] = "-1"
        if not empty_rle(row[args.rle_column]):
            final_nonempty += 1

    Path(args.output_submission).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_submission).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[Gate summary]")
    print(f"  rows                      : {len(rows)}")
    print(f"  missing classifier probs   : {missing}")
    print(f"  originally non-empty RLE   : {originally_nonempty}")
    print(f"  gated non-empty -> empty   : {gated_nonempty_to_empty}")
    print(f"  final non-empty RLE        : {final_nonempty}")
    print(f"  output                     : {args.output_submission}")


if __name__ == "__main__":
    main()
