# scripts/evaluate/merge_oof_cls_predictions.py
# -*- coding: utf-8 -*-
"""Merge per-fold validation predictions into one OOF classifier CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge OOF classification gate predictions.")
    parser.add_argument("--input-glob", type=str, required=True, help="Example: outputs/experiments/cls_gate_5fold/fold_*/val_predictions_best.csv")
    parser.add_argument("--output-csv", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        # pathlib does not support absolute glob directly in all cases
        import glob
        paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    rows: List[dict] = []
    seen = set()
    for fold, path in enumerate(paths):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id") or row.get("ImageId")
                if image_id in seen:
                    raise ValueError(f"Duplicate OOF image_id found: {image_id}")
                seen.add(image_id)
                row = dict(row)
                row["image_id"] = image_id
                row.setdefault("fold", str(fold))
                rows.append(row)

    fieldnames = ["fold", "image_id"]
    for k in ["label", "cls_logit", "cls_prob"]:
        if any(k in r for r in rows):
            fieldnames.append(k)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Merged {len(rows)} rows from {len(paths)} files -> {args.output_csv}")


if __name__ == "__main__":
    main()
