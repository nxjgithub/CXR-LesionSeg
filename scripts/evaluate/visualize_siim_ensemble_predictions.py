# scripts/evaluate/visualize_siim_ensemble_predictions.py
# -*- coding: utf-8 -*-
"""
Visualize ensemble SIIM-ACR predictions after threshold + post-processing.

Typical use:
python scripts/evaluate/visualize_siim_ensemble_predictions.py \
  --run-dirs \
    outputs/experiments/siim_acr_unetpp_resnet34_recall_v2/<TIMESTAMP> \
    outputs/experiments/siim_acr_unetpp_resnet50_recall_v1/<TIMESTAMP> \
  --split val \
  --threshold 0.08 \
  --min-component-area 150 \
  --min-pred-area 30 \
  --tta hflip \
  --top-k 24 \
  --batch-size 16 \
  --num-workers 8 \
  --device cuda \
  --amp \
  --output-dir outputs/experiments/ensemble_unetpp_r34_r50/visualizations/val_ensemble_thr0p08_minarea150
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
from cxr_lesionseg.models import SMPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ensemble SIIM-ACR predictions.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dirs", type=str, nargs="+")
    group.add_argument("--checkpoints", type=str, nargs="+")

    parser.add_argument("--configs", type=str, nargs="+", default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--min-pred-area", type=int, default=0)
    parser.add_argument("--keep-largest-component", action="store_true")
    parser.add_argument("--tta", type=str, default="none", choices=["none", "hflip"])
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)

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
            raise ValueError("Number of --configs must equal --checkpoints.")

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
    print(f"[Loaded] {checkpoint_path}")
    print(f"model={config.get('model_name')} encoder={config.get('encoder_name')} epoch={ckpt.get('epoch')} best={ckpt.get('best_score', ckpt.get('best_dice'))}")
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

    return DataLoader(
        dataset,
        batch_size=args.batch_size or int(base_config.get("batch_size", 16)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(base_config.get("num_workers", 8)),
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )


def postprocess_mask(mask: np.ndarray, min_component_area: int, min_pred_area: int, keep_largest: bool) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask

    if min_component_area > 0 or keep_largest:
        try:
            import cv2

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                areas = stats[:, cv2.CC_STAT_AREA]
                component_ids = list(range(1, num_labels))
                if keep_largest:
                    largest_id = max(component_ids, key=lambda cid: areas[cid])
                    keep_ids = {largest_id}
                else:
                    keep_ids = set(component_ids)
                if min_component_area > 0:
                    keep_ids = {cid for cid in keep_ids if int(areas[cid]) >= int(min_component_area)}
                mask = np.isin(labels, list(keep_ids)).astype(np.uint8) if keep_ids else np.zeros_like(mask)
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
                mask = np.isin(labels, list(keep_ids)).astype(np.uint8) if keep_ids else np.zeros_like(mask)

    if int(mask.sum()) < int(min_pred_area):
        mask = np.zeros_like(mask, dtype=np.uint8)
    return mask.astype(np.uint8)


def per_image_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> Dict[str, float | bool]:
    p = pred.astype(np.float32)
    t = target.astype(np.float32)
    pred_area = float(p.sum())
    gt_area = float(t.sum())
    if pred_area + gt_area == 0:
        dice = 1.0
        iou = 1.0
    else:
        tp = float((p * t).sum())
        fp = float((p * (1.0 - t)).sum())
        fn = float(((1.0 - p) * t).sum())
        dice = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
        iou = float((tp + eps) / (tp + fp + fn + eps))
    return {
        "dice": dice,
        "iou": iou,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "is_positive": gt_area > 0,
        "is_pred_positive": pred_area > 0,
    }


def denormalize_for_display(image: np.ndarray) -> np.ndarray:
    # Input shape: C,H,W. The project dataset normalizes images; for visualization we rescale robustly.
    if image.ndim == 3:
        if image.shape[0] == 1:
            img = image[0]
        else:
            img = np.transpose(image, (1, 2, 0))
    else:
        img = image
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = (img - lo) / max(hi - lo, 1e-6)
    return np.clip(img, 0.0, 1.0)


def overlay_rgb(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = denormalize_for_display(image)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        if base.shape[-1] == 1:
            rgb = np.repeat(base, 3, axis=-1)
        else:
            rgb = base[..., :3].copy()

    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)

    # Green = GT, Red = prediction, Yellow = overlap.
    out = rgb.copy()
    out[gt_bool] = 0.55 * out[gt_bool] + 0.45 * np.array([0.0, 1.0, 0.0])
    out[pred_bool] = 0.55 * out[pred_bool] + 0.45 * np.array([1.0, 0.0, 0.0])
    both = gt_bool & pred_bool
    out[both] = 0.45 * out[both] + 0.55 * np.array([1.0, 1.0, 0.0])
    return np.clip(out, 0.0, 1.0)


def save_record_figure(record: Dict, out_path: Path) -> None:
    image = record["image"]
    gt = record["gt"]
    pred = record["pred"]
    prob = record["prob"]

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(denormalize_for_display(image), cmap="gray")
    ax.set_title("Image")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(gt, cmap="gray")
    ax.set_title(f"GT area={int(record['gt_area'])}")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(pred, cmap="gray")
    ax.set_title(f"Pred area={int(record['pred_area'])}")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(overlay_rgb(image, gt, pred))
    ax.imshow(prob, cmap="magma", alpha=0.22)
    ax.set_title(f"Overlay | Dice={record['dice']:.4f}, IoU={record['iou']:.4f}")
    ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def predict_ensemble_probs(models: List[torch.nn.Module], images: torch.Tensor, amp: bool, device: torch.device, tta: str) -> torch.Tensor:
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


def get_ids_from_batch(batch: Dict, batch_size: int, global_start: int) -> List[str]:
    for key in ["image_id", "id", "case_id", "uid", "path", "image_path"]:
        if key in batch:
            vals = batch[key]
            if isinstance(vals, (list, tuple)):
                return [str(v) for v in vals]
            try:
                return [str(v) for v in list(vals)]
            except Exception:
                pass
    return [f"sample_{global_start + i:06d}" for i in range(batch_size)]


def mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


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
    loader = build_loader(base_config, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["high_dice_positive", "low_dice_positive", "false_positive_negative", "false_negative_positive", "true_negative"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    seen = 0
    print("=" * 100)
    print("[Visualize]")
    print(f"threshold          : {args.threshold}")
    print(f"min_component_area : {args.min_component_area}")
    print(f"min_pred_area      : {args.min_pred_area}")
    print(f"keep_largest       : {args.keep_largest_component}")
    print(f"tta                : {args.tta}")
    print(f"output_dir         : {output_dir}")
    print("=" * 100)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        probs = predict_ensemble_probs(models, images, args.amp, device, args.tta)

        images_np = batch["image"].detach().cpu().numpy()
        masks_np = (masks.detach().cpu().numpy()[:, 0] >= 0.5).astype(np.uint8)
        probs_np = probs.detach().cpu().numpy()[:, 0]
        batch_size = probs_np.shape[0]
        ids = get_ids_from_batch(batch, batch_size, seen)

        for i in range(batch_size):
            raw_pred = (probs_np[i] >= args.threshold).astype(np.uint8)
            pred = postprocess_mask(raw_pred, args.min_component_area, args.min_pred_area, args.keep_largest_component)
            gt = masks_np[i]
            m = per_image_metrics(pred, gt)
            records.append(
                {
                    "index": seen + i,
                    "image_id": ids[i],
                    "dice": float(m["dice"]),
                    "iou": float(m["iou"]),
                    "pred_area": float(m["pred_area"]),
                    "gt_area": float(m["gt_area"]),
                    "is_positive": bool(m["is_positive"]),
                    "is_pred_positive": bool(m["is_pred_positive"]),
                    "image": images_np[i],
                    "gt": gt,
                    "pred": pred,
                    "prob": probs_np[i],
                }
            )

        seen += batch_size
        if step % 20 == 0:
            print(f"[Evaluate] step={step}, seen={seen}")

    positives = [r for r in records if r["is_positive"]]
    negatives = [r for r in records if not r["is_positive"]]
    fp_negatives = [r for r in negatives if r["is_pred_positive"]]
    tn = [r for r in negatives if not r["is_pred_positive"]]
    fn_positives = [r for r in positives if not r["is_pred_positive"]]

    summary = {
        "num_images": len(records),
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_false_positive_negative": len(fp_negatives),
        "num_true_negative": len(tn),
        "num_false_negative_positive": len(fn_positives),
        "negative_fp_rate": len(fp_negatives) / max(len(negatives), 1),
        "positive_image_recall": (len(positives) - len(fn_positives)) / max(len(positives), 1),
        "mean_dice_all": mean_or_zero([float(r["dice"]) for r in records]),
        "mean_dice_positive": mean_or_zero([float(r["dice"]) for r in positives]),
        "mean_dice_negative": mean_or_zero([float(r["dice"]) for r in negatives]),
        "mean_iou_positive": mean_or_zero([float(r["iou"]) for r in positives]),
        "mean_pred_area_positive": mean_or_zero([float(r["pred_area"]) for r in positives]),
        "mean_pred_area_negative": mean_or_zero([float(r["pred_area"]) for r in negatives]),
        "threshold": args.threshold,
        "min_component_area": args.min_component_area,
        "min_pred_area": args.min_pred_area,
        "keep_largest_component": args.keep_largest_component,
        "tta": args.tta,
        "num_models": len(models),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Records CSV without heavy arrays.
    csv_rows = []
    for r in records:
        csv_rows.append({k: r[k] for k in ["index", "image_id", "dice", "iou", "pred_area", "gt_area", "is_positive", "is_pred_positive"]})
    with (output_dir / "records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    selections = {
        "high_dice_positive": sorted(positives, key=lambda r: r["dice"], reverse=True)[: args.top_k],
        "low_dice_positive": sorted(positives, key=lambda r: r["dice"])[: args.top_k],
        "false_positive_negative": sorted(fp_negatives, key=lambda r: r["pred_area"], reverse=True)[: args.top_k],
        "false_negative_positive": sorted(fn_positives, key=lambda r: r["gt_area"], reverse=True)[: args.top_k],
        "true_negative": tn[: args.top_k],
    }

    for category, selected in selections.items():
        for rank, r in enumerate(selected, start=1):
            safe_id = str(r["image_id"]).replace("/", "_").replace(" ", "_")
            filename = f"{rank:03d}_dice{float(r['dice']):.4f}_pred{int(r['pred_area'])}_gt{int(r['gt_area'])}_{safe_id}.png"
            save_record_figure(r, output_dir / category / filename)

    print("=" * 100)
    print("[Summary]")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 100)
    print(f"Saved summary: {output_dir / 'summary.json'}")
    print(f"Saved records: {output_dir / 'records.csv'}")
    print(f"Saved figures: {output_dir}")


if __name__ == "__main__":
    main()
