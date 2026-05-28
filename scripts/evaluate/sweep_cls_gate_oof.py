# scripts/evaluate/sweep_cls_gate_oof.py
# -*- coding: utf-8 -*-
"""Sweep classifier gate threshold on OOF predictions.

This script evaluates the image-level gate itself. It does not evaluate mask Dice.
Use it to pick a safe classifier threshold before combining it with segmentation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None


def parse_float_list(value: str) -> List[float]:
    if ":" in value:
        start, stop, step = [float(x) for x in value.split(":")]
        vals = []
        x = start
        while x <= stop + 1e-12:
            vals.append(round(x, 8))
            x += step
        return vals
    return [float(x) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sweep OOF classification gate threshold.")
    parser.add_argument("--oof-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default="0.01:0.99:0.01", help="range start:stop:step or comma list")
    parser.add_argument("--target-negative-fp-rate", type=float, default=0.05)
    parser.add_argument("--prob-column", type=str, default="cls_prob")
    parser.add_argument("--label-column", type=str, default="label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with Path(args.oof_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty OOF CSV: {args.oof_csv}")
    labels = np.asarray([int(float(r[args.label_column])) for r in rows], dtype=np.int32)
    probs = np.asarray([float(r[args.prob_column]) for r in rows], dtype=np.float32)

    auc = float(roc_auc_score(labels, probs)) if roc_auc_score is not None and len(np.unique(labels)) == 2 else float("nan")
    ap = float(average_precision_score(labels, probs)) if average_precision_score is not None and len(np.unique(labels)) == 2 else float("nan")

    out = []
    for thr in parse_float_list(args.thresholds):
        pred = (probs >= thr).astype(np.int32)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        eps = 1e-12
        precision = tp / max(eps, tp + fp)
        recall = tp / max(eps, tp + fn)
        specificity = tn / max(eps, tn + fp)
        negative_fp_rate = fp / max(eps, tn + fp)
        f2 = (5 * precision * recall) / max(eps, 4 * precision + recall)
        out.append({
            "threshold": thr,
            "auc": auc,
            "ap": ap,
            "acc": (tp + tn) / max(1, len(labels)),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "negative_fp_rate": negative_fp_rate,
            "predicted_positive_rate": float(pred.mean()),
            "f2": f2,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    feasible = [r for r in out if r["negative_fp_rate"] <= args.target_negative_fp_rate]
    if feasible:
        best = max(feasible, key=lambda r: (r["recall"], r["f2"], r["threshold"]))
        print(f"[Best under negative_fp_rate<={args.target_negative_fp_rate:.3f}] {best}")
    best_f2 = max(out, key=lambda r: r["f2"])
    print(f"[Best F2 overall] {best_f2}")
    print(f"[OK] Wrote sweep: {args.output_csv}")


if __name__ == "__main__":
    main()
