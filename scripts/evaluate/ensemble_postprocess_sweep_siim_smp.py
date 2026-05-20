# scripts/evaluate/ensemble_postprocess_sweep_siim_smp.py
# -*- coding: utf-8 -*-
"""
Ensemble threshold + post-process sweep for SIIM-ACR SMP segmentation models.

Typical use:
python scripts/evaluate/ensemble_postprocess_sweep_siim_smp.py \
  --run-dirs \
    outputs/experiments/siim_acr_unetpp_resnet34_recall_v2/<TIMESTAMP> \
    outputs/experiments/siim_acr_unetpp_resnet50_recall_v1/<TIMESTAMP> \
  --split val \
  --thresholds 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.15 0.20 0.25 0.30 \
  --min-component-areas 0 50 100 150 200 300 500 \
  --min-pred-areas 0 20 30 50 80 100 \
  --keep-largest-options 0 \
  --tta hflip \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --amp \
  --output-csv outputs/experiments/ensemble_unetpp_r34_r50/val_ensemble_postprocess_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.models import SMPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble threshold/post-process sweep for SIIM-ACR SMP models."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dirs",
        type=str,
        nargs="+",
        help="Experiment run dirs. Each dir should contain config.json and checkpoints/best_model.pth.",
    )
    group.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        help="Checkpoint paths. Must be used together with --configs.",
    )

    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Config paths. Required when --checkpoints is used. Ignored when --run-dirs is used.",
    )

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-csv", type=str, required=True)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    )
    parser.add_argument(
        "--min-component-areas",
        type=int,
        nargs="+",
        default=[0, 50, 100, 150, 200, 300, 500],
        help="Remove connected components smaller than this area. Pixel units after resizing.",
    )
    parser.add_argument(
        "--min-pred-areas",
        type=int,
        nargs="+",
        default=[0, 20, 30, 50, 80, 100],
        help="Set the whole mask to empty if final predicted area is smaller than this value.",
    )
    parser.add_argument(
        "--keep-largest-options",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
        help="0/1. Usually keep 0 first for pneumothorax because multiple separated regions may exist.",
    )
    parser.add_argument(
        "--tta",
        type=str,
        default="none",
        choices=["none", "hflip"],
        help="Test-time augmentation. hflip averages original and horizontal-flip predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)

    return parser.parse_args()


def safe_torch_load(path: str | Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_inputs(args: argparse.Namespace) -> Tuple[List[Path], List[Path]]:
    if args.run_dirs is not None:
        run_dirs = [Path(p) for p in args.run_dirs]
        checkpoints = [p / "checkpoints" / "best_model.pth" for p in run_dirs]
        configs = [p / "config.json" for p in run_dirs]
    else:
        checkpoints = [Path(p) for p in args.checkpoints]
        if args.configs is None:
            raise ValueError("--configs is required when --checkpoints is used.")
        configs = [Path(p) for p in args.configs]
        if len(configs) != len(checkpoints):
            raise ValueError(
                f"Number of --configs must equal --checkpoints. "
                f"Got configs={len(configs)}, checkpoints={len(checkpoints)}"
            )

    for p in checkpoints + configs:
        if not p.exists():
            raise FileNotFoundError(str(p))

    return checkpoints, configs


def merged_config(config_path: Path, ckpt: Dict) -> Dict:
    cfg: Dict = {}
    if isinstance(ckpt.get("config"), dict):
        cfg.update(ckpt["config"])
    cfg.update(load_json(config_path))
    return cfg


def build_model(config: Dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = safe_torch_load(checkpoint_path, device)

    model = SMPModel(
        model_name=str(config.get("model_name", "unetplusplus")),
        encoder_name=str(config.get("encoder_name", "resnet34")),
        encoder_weights=None,
        in_channels=int(config.get("in_channels", 3)),
        classes=1,
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print("-" * 100)
    print("[Loaded model]")
    print(f"checkpoint   : {checkpoint_path}")
    print(f"epoch        : {ckpt.get('epoch')}")
    print(f"best_score   : {ckpt.get('best_score', ckpt.get('best_dice'))}")
    print(f"model_name   : {config.get('model_name')}")
    print(f"encoder_name : {config.get('encoder_name')}")
    print(f"in_channels  : {config.get('in_channels')}")

    return model


def build_loader(base_config: Dict, args: argparse.Namespace) -> DataLoader:
    dataset = SIIMACRDataset(
        processed_root=base_config.get("processed_root", "data/processed/siim_acr_pneumothorax"),
        split_root=base_config.get("split_root", "data/splits/siim_acr_pneumothorax"),
        split=args.split,
        image_size=int(base_config.get("image_size", 512)),
        in_channels=int(base_config.get("in_channels", 3)),
        normalize=True,
    )

    if args.max_samples is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(base_config.get("batch_size", 16)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(base_config.get("num_workers", 8)),
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    print("=" * 100)
    print("[Dataset]")
    print(f"split       : {args.split}")
    print(f"num_images  : {len(dataset)}")
    print(f"image_size  : {base_config.get('image_size', 512)}")
    print(f"in_channels : {base_config.get('in_channels', 3)}")
    print("=" * 100)

    return loader


def postprocess_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    min_pred_area: int = 0,
    keep_largest: bool = False,
) -> np.ndarray:
    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        return mask

    if min_component_area > 0 or keep_largest:
        try:
            import cv2

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                pass
            else:
                areas = stats[:, cv2.CC_STAT_AREA]
                component_ids = list(range(1, num_labels))
                if keep_largest:
                    largest_id = max(component_ids, key=lambda cid: areas[cid])
                    keep_ids = {largest_id}
                else:
                    keep_ids = set(component_ids)

                if min_component_area > 0:
                    keep_ids = {cid for cid in keep_ids if int(areas[cid]) >= int(min_component_area)}

                if keep_ids:
                    mask = np.isin(labels, list(keep_ids)).astype(np.uint8)
                else:
                    mask = np.zeros_like(mask, dtype=np.uint8)
        except Exception:
            from scipy import ndimage

            labels, num_foreground = ndimage.label(mask)
            num_labels = num_foreground + 1
            if num_labels > 1:
                areas = np.bincount(labels.reshape(-1))
                component_ids = list(range(1, num_labels))
                if keep_largest:
                    largest_id = max(component_ids, key=lambda cid: areas[cid])
                    keep_ids = {largest_id}
                else:
                    keep_ids = set(component_ids)

                if min_component_area > 0:
                    keep_ids = {cid for cid in keep_ids if int(areas[cid]) >= int(min_component_area)}

                if keep_ids:
                    mask = np.isin(labels, list(keep_ids)).astype(np.uint8)
                else:
                    mask = np.zeros_like(mask, dtype=np.uint8)

    if int(mask.sum()) < int(min_pred_area):
        mask = np.zeros_like(mask, dtype=np.uint8)

    return mask


def empty_stats() -> Dict[str, float]:
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}


def update_global_stats(stats: Dict[str, float], pred: np.ndarray, target: np.ndarray) -> None:
    p = pred.astype(np.float32).reshape(-1)
    t = target.astype(np.float32).reshape(-1)
    stats["tp"] += float((p * t).sum())
    stats["fp"] += float((p * (1.0 - t)).sum())
    stats["fn"] += float(((1.0 - p) * t).sum())
    stats["tn"] += float(((1.0 - p) * (1.0 - t)).sum())


def metrics_from_stats(stats: Dict[str, float], eps: float = 1e-7) -> Dict[str, float]:
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    return {
        "pixel_dice": float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)),
        "pixel_iou": float((tp + eps) / (tp + fp + fn + eps)),
        "pixel_precision": float((tp + eps) / (tp + fp + eps)),
        "pixel_recall": float((tp + eps) / (tp + fn + eps)),
        "pixel_specificity": float((tn + eps) / (tn + fp + eps)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def per_image_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    p = pred.astype(np.float32)
    t = target.astype(np.float32)
    pred_area = float(p.sum())
    target_area = float(t.sum())

    if pred_area + target_area == 0:
        dice = 1.0
        iou = 1.0
    else:
        tp = float((p * t).sum())
        fp = float((p * (1.0 - t)).sum())
        fn = float(((1.0 - p) * t).sum())
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "pred_area": pred_area,
        "target_area": target_area,
        "is_positive": bool(target_area > 0),
        "is_pred_positive": bool(pred_area > 0),
    }


def mean_or_zero(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


@torch.no_grad()
def predict_ensemble_probs(
    models: List[torch.nn.Module],
    images: torch.Tensor,
    amp: bool,
    device: torch.device,
    tta: str,
) -> torch.Tensor:
    probs_sum: Optional[torch.Tensor] = None
    n = 0

    with torch.amp.autocast(device_type="cuda", enabled=bool(amp and device.type == "cuda")):
        for model in models:
            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            n += 1

            if tta == "hflip":
                flip_images = torch.flip(images, dims=[3])
                flip_logits = model(flip_images)
                flip_probs = torch.sigmoid(flip_logits)
                flip_probs = torch.flip(flip_probs, dims=[3])
                probs_sum = probs_sum + flip_probs
                n += 1

    return probs_sum / max(n, 1)


def build_sweep_configs(args: argparse.Namespace) -> List[Dict]:
    configs: List[Dict] = []
    for thr in args.thresholds:
        for min_component_area in args.min_component_areas:
            for min_pred_area in args.min_pred_areas:
                for keep_largest in args.keep_largest_options:
                    configs.append(
                        {
                            "threshold": float(thr),
                            "min_component_area": int(min_component_area),
                            "min_pred_area": int(min_pred_area),
                            "keep_largest_component": bool(int(keep_largest)),
                            "records": [],
                            "stats": empty_stats(),
                        }
                    )
    return configs


def summarize_config(cfg: Dict, num_images: int) -> Dict[str, float | int | str]:
    records = cfg["records"]
    positives = [r for r in records if r["is_positive"]]
    negatives = [r for r in records if not r["is_positive"]]

    false_positive_negative = sum(1 for r in negatives if r["is_pred_positive"])
    true_negative = sum(1 for r in negatives if not r["is_pred_positive"])
    false_negative_positive = sum(1 for r in positives if not r["is_pred_positive"])
    detected_positive = sum(1 for r in positives if r["is_pred_positive"])

    summary: Dict[str, float | int | str] = {
        "threshold": cfg["threshold"],
        "min_component_area": cfg["min_component_area"],
        "min_pred_area": cfg["min_pred_area"],
        "keep_largest_component": int(cfg["keep_largest_component"]),
        "num_images": num_images,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_false_positive_negative": false_positive_negative,
        "num_true_negative": true_negative,
        "num_false_negative_positive": false_negative_positive,
        "num_detected_positive": detected_positive,
        "negative_fp_rate": false_positive_negative / max(len(negatives), 1),
        "positive_image_recall": detected_positive / max(len(positives), 1),
        "mean_dice_all": mean_or_zero(r["dice"] for r in records),
        "mean_dice_positive": mean_or_zero(r["dice"] for r in positives),
        "mean_dice_negative": mean_or_zero(r["dice"] for r in negatives),
        "mean_iou_positive": mean_or_zero(r["iou"] for r in positives),
        "mean_pred_area_positive": mean_or_zero(r["pred_area"] for r in positives),
        "mean_pred_area_negative": mean_or_zero(r["pred_area"] for r in negatives),
    }
    summary.update(metrics_from_stats(cfg["stats"]))
    return summary


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    checkpoints, config_paths = resolve_model_inputs(args)

    configs: List[Dict] = []
    models: List[torch.nn.Module] = []
    for ckpt_path, cfg_path in zip(checkpoints, config_paths):
        ckpt = safe_torch_load(ckpt_path, device)
        cfg = merged_config(cfg_path, ckpt)
        configs.append(cfg)
        models.append(build_model(cfg, ckpt_path, device))

    base_config = configs[0]
    for idx, cfg in enumerate(configs[1:], start=2):
        for key in ["processed_root", "split_root", "image_size", "in_channels"]:
            if str(cfg.get(key)) != str(base_config.get(key)):
                print(f"[Warning] model {idx} config differs on {key}: {cfg.get(key)} vs {base_config.get(key)}")

    loader = build_loader(base_config, args)
    sweep_configs = build_sweep_configs(args)

    print("=" * 100)
    print("[Sweep]")
    print(f"num_models             : {len(models)}")
    print(f"tta                    : {args.tta}")
    print(f"num_sweep_configs      : {len(sweep_configs)}")
    print(f"output_csv             : {args.output_csv}")
    print("=" * 100)

    seen = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        probs = predict_ensemble_probs(
            models=models,
            images=images,
            amp=args.amp,
            device=device,
            tta=args.tta,
        )

        probs_np = probs.detach().cpu().numpy()[:, 0]
        masks_np = (masks.detach().cpu().numpy()[:, 0] >= 0.5).astype(np.uint8)

        batch_size = probs_np.shape[0]
        seen += batch_size

        for cfg in sweep_configs:
            thr = float(cfg["threshold"])
            min_component_area = int(cfg["min_component_area"])
            min_pred_area = int(cfg["min_pred_area"])
            keep_largest = bool(cfg["keep_largest_component"])

            for i in range(batch_size):
                pred = (probs_np[i] >= thr).astype(np.uint8)
                pred = postprocess_mask(
                    pred,
                    min_component_area=min_component_area,
                    min_pred_area=min_pred_area,
                    keep_largest=keep_largest,
                )
                target = masks_np[i]

                update_global_stats(cfg["stats"], pred, target)
                cfg["records"].append(per_image_metrics(pred, target))

        if step % 10 == 0:
            print(f"[Evaluate] step={step}, seen={seen}")

    rows = [summarize_config(cfg, seen) for cfg in sweep_configs]
    rows = sorted(
        rows,
        key=lambda r: (
            float(r["mean_dice_all"]),
            float(r["mean_dice_positive"]),
            -float(r["negative_fp_rate"]),
        ),
        reverse=True,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("[Top 20 by mean_dice_all]")
    for row in rows[:20]:
        print(
            f"thr={row['threshold']:<5} "
            f"min_comp={row['min_component_area']:<4} "
            f"min_pred={row['min_pred_area']:<4} "
            f"keep_largest={row['keep_largest_component']} "
            f"mean_all={float(row['mean_dice_all']):.4f} "
            f"pos_dice={float(row['mean_dice_positive']):.4f} "
            f"neg_fp={float(row['negative_fp_rate']):.4f} "
            f"pos_img_recall={float(row['positive_image_recall']):.4f}"
        )
    print("=" * 100)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
