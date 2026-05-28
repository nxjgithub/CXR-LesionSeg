# scripts/preprocess/make_siim_5fold_splits.py
# -*- coding: utf-8 -*-
"""Create stratified 5-fold image-level splits for SIIM-ACR.

Input:
  data/processed/siim_acr_pneumothorax/mask_summary.csv

Output:
  data/splits/siim_acr_pneumothorax_5fold/fold_0_train.txt
  data/splits/siim_acr_pneumothorax_5fold/fold_0_val.txt
  ...
  folds_summary.csv

The split is stratified by image-level label: has_mask = 1/0.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create SIIM-ACR stratified 5-fold splits.")
    parser.add_argument("--processed-root", type=str, default="data/processed/siim_acr_pneumothorax")
    parser.add_argument("--output-root", type=str, default="data/splits/siim_acr_pneumothorax_5fold")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-column", type=str, default="ImageId")
    parser.add_argument("--label-column", type=str, default="has_mask")
    return parser.parse_args()


def read_labels(mask_summary_path: Path, id_col: str, label_col: str) -> List[Tuple[str, int]]:
    if not mask_summary_path.exists():
        raise FileNotFoundError(f"mask_summary.csv not found: {mask_summary_path}")

    rows: List[Tuple[str, int]] = []
    with mask_summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {mask_summary_path}")
        cols = {c.strip().lstrip("\ufeff") for c in reader.fieldnames}
        if id_col not in cols:
            # common fallback from earlier code
            if "image_id" in cols:
                id_col = "image_id"
            else:
                raise ValueError(f"Missing id column {id_col!r}. Available columns: {sorted(cols)}")
        if label_col not in cols:
            raise ValueError(f"Missing label column {label_col!r}. Available columns: {sorted(cols)}")

        for row in reader:
            image_id = str(row[id_col]).strip()
            if not image_id:
                continue
            raw = str(row[label_col]).strip().lower()
            label = 1 if raw in {"1", "true", "yes", "y"} else 0
            rows.append((image_id, label))

    if not rows:
        raise ValueError(f"No rows loaded from {mask_summary_path}")
    return rows


def stratified_folds(items: List[Tuple[str, int]], num_folds: int, seed: int) -> List[List[Tuple[str, int]]]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {0: [], 1: []}
    for item in items:
        by_label[item[1]].append(item)

    folds: List[List[Tuple[str, int]]] = [[] for _ in range(num_folds)]
    for label, label_items in by_label.items():
        rng.shuffle(label_items)
        for idx, item in enumerate(label_items):
            folds[idx % num_folds].append(item)

    for fold in folds:
        rng.shuffle(fold)
    return folds


def write_ids(path: Path, ids: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for image_id in ids:
            f.write(f"{image_id}\n")


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    items = read_labels(processed_root / "mask_summary.csv", args.id_column, args.label_column)
    folds = stratified_folds(items, args.num_folds, args.seed)

    all_ids = [image_id for image_id, _ in items]
    write_ids(output_root / "all_train.txt", sorted(all_ids))

    summary_rows = []
    for fold_idx in range(args.num_folds):
        val_items = folds[fold_idx]
        train_items = [item for i, fold in enumerate(folds) if i != fold_idx for item in fold]

        train_ids = sorted([image_id for image_id, _ in train_items])
        val_ids = sorted([image_id for image_id, _ in val_items])

        write_ids(output_root / f"fold_{fold_idx}_train.txt", train_ids)
        write_ids(output_root / f"fold_{fold_idx}_val.txt", val_ids)

        train_counter = Counter(label for _, label in train_items)
        val_counter = Counter(label for _, label in val_items)
        summary_rows.append({
            "fold": fold_idx,
            "train_total": len(train_items),
            "train_positive": train_counter[1],
            "train_negative": train_counter[0],
            "val_total": len(val_items),
            "val_positive": val_counter[1],
            "val_negative": val_counter[0],
            "val_positive_ratio": val_counter[1] / max(1, len(val_items)),
        })

    with (output_root / "folds_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Wrote {args.num_folds}-fold splits to: {output_root}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
