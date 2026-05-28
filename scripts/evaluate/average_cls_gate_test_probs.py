# scripts/evaluate/average_cls_gate_test_probs.py
# -*- coding: utf-8 -*-
"""Average five fold classifier probabilities on official test images."""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Average fold classifier probabilities.")
    parser.add_argument("--input-glob", type=str, required=True, help="Example: outputs/.../fold_*/test_predictions.csv")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--prob-column", type=str, default="cls_prob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")
    probs: Dict[str, List[float]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id") or row.get("ImageId")
                probs[image_id].append(float(row[args.prob_column]))

    rows = []
    expected = len(paths)
    for image_id in sorted(probs):
        values = probs[image_id]
        if len(values) != expected:
            raise ValueError(f"image_id={image_id} has {len(values)} predictions, expected {expected}")
        rows.append({
            "image_id": image_id,
            "cls_prob": float(np.mean(values)),
            "cls_prob_std": float(np.std(values)),
            "num_folds": len(values),
        })

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "cls_prob", "cls_prob_std", "num_folds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote averaged probabilities: {args.output_csv} rows={len(rows)} folds={expected}")


if __name__ == "__main__":
    main()
