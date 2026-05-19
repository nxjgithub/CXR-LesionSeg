# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset

try:
    from cxr_lesionseg.models import SMPModel
except ImportError:
    from cxr_lesionseg.models.smp_models import SMPModel


def parse_args():
    parser = argparse.ArgumentParser("Postprocess sweep for SIIM-ACR SMP model")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-csv", type=str, required=True)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.25],
    )
    parser.add_argument(
        "--min-component-areas",
        type=int,
        nargs="+",
        default=[0, 100, 300, 500, 1000],
    )
    parser.add_argument(
        "--keep-largest-options",
        type=int,
        nargs="+",
        default=[0, 1],
        help="0=false, 1=true",
    )

    parser.add_argument("--min-pred-area", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)

    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_config(config_path: str, ckpt: Dict) -> Dict:
    config = {}

    if "config" in ckpt and isinstance(ckpt["config"], dict):
        config.update(ckpt["config"])

    with open(config_path, "r", encoding="utf-8") as f:
        file_config = json.load(f)

    config.update(file_config)
    return config


def build_model(config: Dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = safe_torch_load(checkpoint_path, device=device)

    model = SMPModel(
        model_name=str(config.get("model_name", "unetplusplus")),
        encoder_name=str(config.get("encoder_name", "resnet34")),
        encoder_weights=None,
        in_channels=int(config.get("in_channels", 3)),
        classes=1,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("=" * 100)
    print("[Checkpoint]")
    print(f"checkpoint   : {checkpoint_path}")
    print(f"epoch        : {ckpt.get('epoch')}")
    print(f"best_score   : {ckpt.get('best_score', ckpt.get('best_dice'))}")
    print(f"model_name   : {config.get('model_name')}")
    print(f"encoder_name : {config.get('encoder_name')}")
    print(f"in_channels  : {config.get('in_channels')}")
    print("=" * 100)

    return model


def build_loader(config: Dict, args) -> DataLoader:
    dataset = SIIMACRDataset(
        processed_root=config.get("processed_root", "data/processed/siim_acr_pneumothorax"),
        split_root=config.get("split_root", "data/splits/siim_acr_pneumothorax"),
        split=args.split,
        image_size=int(config.get("image_size", 512)),
        in_channels=int(config.get("in_channels", 3)),
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(config.get("num_workers", 8)),
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    print("=" * 100)
    print("[Dataset]")
    print(f"split       : {args.split}")
    print(f"num_images  : {len(dataset)}")
    print(f"image_size  : {config.get('image_size', 512)}")
    print(f"in_channels : {config.get('in_channels', 3)}")
    print("=" * 100)

    return loader


def postprocess_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    keep_largest: bool = False,
) -> np.ndarray:
    """
    mask: [H, W], uint8, values 0/1
    """

    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        return mask

    if min_component_area <= 0 and not keep_largest:
        return mask

    try:
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            return mask

        areas = stats[:, cv2.CC_STAT_AREA]

    except Exception:
        from scipy import ndimage

        labels, num_foreground = ndimage.label(mask)
        num_labels = num_foreground + 1

        if num_labels <= 1:
            return mask

        areas = np.bincount(labels.reshape(-1))

    component_ids = list(range(1, num_labels))

    if keep_largest:
        largest_id = max(component_ids, key=lambda cid: areas[cid])
        keep_ids = {largest_id}
    else:
        keep_ids = set(component_ids)

    if min_component_area > 0:
        keep_ids = {
            cid for cid in keep_ids
            if int(areas[cid]) >= int(min_component_area)
        }

    if not keep_ids:
        return np.zeros_like(mask, dtype=np.uint8)

    out = np.isin(labels, list(keep_ids)).astype(np.uint8)
    return out


def compute_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    pred = pred.astype(np.float32).reshape(-1)
    target = target.astype(np.float32).reshape(-1)

    tp = float((pred * target).sum())
    fp = float((pred * (1.0 - target)).sum())
    fn = float(((1.0 - pred) * target).sum())
    tn = float(((1.0 - pred) * (1.0 - target)).sum())

    pred_area = float(pred.sum())
    target_area = float(target.sum())

    if pred_area + target_area == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pred_area": pred_area,
        "target_area": target_area,
    }


def mean_value(rows: List[Dict], key: str) -> float:
    if len(rows) == 0:
        return 0.0
    return float(np.mean([float(r[key]) for r in rows]))


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    ckpt = safe_torch_load(args.checkpoint, device=device)
    config = load_config(args.config, ckpt)

    model = build_model(config, args.checkpoint, device)
    loader = build_loader(config, args)

    configs = []

    for threshold in args.thresholds:
        for min_area in args.min_component_areas:
            for keep_largest in args.keep_largest_options:
                configs.append(
                    {
                        "threshold": float(threshold),
                        "min_component_area": int(min_area),
                        "keep_largest_component": bool(int(keep_largest)),
                        "records": [],
                    }
                )

    autocast_enabled = bool(args.amp and device.type == "cuda")
    seen = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            logits = model(images)
            probs = torch.sigmoid(logits)

        probs_np = probs[:, 0].detach().cpu().numpy()
        masks_np = masks[:, 0].detach().cpu().numpy()

        batch_size = probs_np.shape[0]

        for i in range(batch_size):
            global_index = seen + i

            if args.max_samples is not None and global_index >= args.max_samples:
                break

            target = (masks_np[i] >= 0.5).astype(np.uint8)
            target_area = int(target.sum())

            for cfg in configs:
                pred = (probs_np[i] >= cfg["threshold"]).astype(np.uint8)

                pred = postprocess_mask(
                    pred,
                    min_component_area=cfg["min_component_area"],
                    keep_largest=cfg["keep_largest_component"],
                )

                metrics = compute_metrics(pred, target)

                pred_area = int(metrics["pred_area"])

                is_positive = target_area > 0
                is_negative = target_area == 0
                is_fp_negative = bool(is_negative and pred_area >= args.min_pred_area)
                is_true_negative = bool(is_negative and pred_area < args.min_pred_area)

                cfg["records"].append(
                    {
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "pred_area": pred_area,
                        "target_area": target_area,
                        "is_positive": int(is_positive),
                        "is_negative": int(is_negative),
                        "is_fp_negative": int(is_fp_negative),
                        "is_true_negative": int(is_true_negative),
                    }
                )

        seen += batch_size

        if step % 20 == 0:
            print(f"[Evaluate] step={step}, seen={seen}")

        if args.max_samples is not None and seen >= args.max_samples:
            break

    rows = []

    for cfg in configs:
        records = cfg["records"]

        positive = [r for r in records if r["is_positive"] == 1]
        negative = [r for r in records if r["is_negative"] == 1]
        fp_negative = [r for r in negative if r["is_fp_negative"] == 1]
        true_negative = [r for r in negative if r["is_true_negative"] == 1]

        row = {
            "threshold": cfg["threshold"],
            "min_component_area": cfg["min_component_area"],
            "keep_largest_component": int(cfg["keep_largest_component"]),
            "num_images": len(records),
            "num_positive": len(positive),
            "num_negative": len(negative),
            "num_false_positive_negative": len(fp_negative),
            "num_true_negative": len(true_negative),
            "negative_fp_rate": len(fp_negative) / max(len(negative), 1),
            "mean_dice_all": mean_value(records, "dice"),
            "mean_dice_positive": mean_value(positive, "dice"),
            "mean_dice_negative": mean_value(negative, "dice"),
            "mean_iou_positive": mean_value(positive, "iou"),
            "mean_pred_area_positive": mean_value(positive, "pred_area"),
            "mean_pred_area_negative": mean_value(negative, "pred_area"),
        }

        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("[Postprocess Sweep Done]")
    print(f"Saved to: {output_csv}")
    print("=" * 100)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            -float(x["mean_dice_positive"]),
            float(x["negative_fp_rate"]),
        ),
    )

    print(
        f"{'thr':>6} {'min_area':>8} {'largest':>8} "
        f"{'pos_dice':>10} {'pos_iou':>10} {'neg_fp':>10} {'neg_area':>10} {'all_dice':>10}"
    )

    for r in rows_sorted[:20]:
        print(
            f"{r['threshold']:6.2f} "
            f"{r['min_component_area']:8d} "
            f"{r['keep_largest_component']:8d} "
            f"{r['mean_dice_positive']:10.4f} "
            f"{r['mean_iou_positive']:10.4f} "
            f"{r['negative_fp_rate']:10.4f} "
            f"{r['mean_pred_area_negative']:10.2f} "
            f"{r['mean_dice_all']:10.4f}"
        )


if __name__ == "__main__":
    main()
