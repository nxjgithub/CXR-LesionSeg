# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset  # noqa: E402

try:
    from cxr_lesionseg.models import SMPModel  # noqa: E402
except ImportError:
    from cxr_lesionseg.models.smp_models import SMPModel  # noqa: E402

EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fast postprocess sweep for SIIM-ACR SMP model")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-csv", type=str, required=True)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    parser.add_argument(
        "--min-component-areas",
        type=int,
        nargs="+",
        default=[0, 50, 100, 150, 200, 300, 500, 800, 1000, 1500],
    )
    parser.add_argument(
        "--min-pred-areas",
        type=int,
        nargs="+",
        default=None,
        help="Set final mask to empty if predicted area is smaller than this value.",
    )
    parser.add_argument(
        "--min-pred-area",
        type=int,
        default=30,
        help="Backward-compatible single value. Used only when --min-pred-areas is not supplied.",
    )
    parser.add_argument(
        "--keep-largest-options",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
    )

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5)

    return parser.parse_args()


def safe_torch_load(path: str | Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_config(config_path: str | Path, ckpt: Dict) -> Dict:
    config: Dict = {}
    if isinstance(ckpt.get("config"), dict):
        config.update(ckpt["config"])
    with Path(config_path).open("r", encoding="utf-8") as f:
        config.update(json.load(f))
    return config


def build_model(config: Dict, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = safe_torch_load(checkpoint_path, device=device)

    model = SMPModel(
        model_name=str(config.get("model_name", "unetplusplus")),
        encoder_name=str(config.get("encoder_name", "resnet34")),
        encoder_weights=None,
        in_channels=int(config.get("in_channels", 3)),
        classes=1,
        decoder_attention_type=config.get("decoder_attention_type", None),
    )

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print("=" * 100, flush=True)
    print("[Checkpoint]", flush=True)
    print(f"checkpoint   : {checkpoint_path}", flush=True)
    print(f"epoch        : {ckpt.get('epoch')}", flush=True)
    print(f"best_score   : {ckpt.get('best_score', ckpt.get('best_dice'))}", flush=True)
    print(f"model_name   : {config.get('model_name')}", flush=True)
    print(f"encoder_name : {config.get('encoder_name')}", flush=True)
    print(f"in_channels  : {config.get('in_channels')}", flush=True)
    print(f"decoder_attn : {config.get('decoder_attention_type')}", flush=True)
    print("=" * 100, flush=True)
    return model


def build_loader(config: Dict, args: argparse.Namespace) -> DataLoader:
    dataset = SIIMACRDataset(
        processed_root=config.get("processed_root", "data/processed/siim_acr_pneumothorax"),
        split_root=config.get("split_root", "data/splits/siim_acr_pneumothorax"),
        split=args.split,
        image_size=int(config.get("image_size", 512)),
        in_channels=int(config.get("in_channels", 3)),
        normalize=True,
    )

    if args.max_samples is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(config.get("num_workers", 8)),
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    print("=" * 100, flush=True)
    print("[Dataset]", flush=True)
    print(f"split       : {args.split}", flush=True)
    print(f"num_images  : {len(dataset)}", flush=True)
    print(f"image_size  : {config.get('image_size', 512)}", flush=True)
    print(f"in_channels : {config.get('in_channels', 3)}", flush=True)
    print("=" * 100, flush=True)
    return loader


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None, List[int]]:
    """Return labels, areas, component ids for a binary mask."""
    if int(mask.sum()) == 0:
        return None, None, []
    try:
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
        component_ids = list(range(1, num_labels))
        return labels, areas, component_ids
    except Exception:
        from scipy import ndimage
        labels, num_foreground = ndimage.label(mask.astype(np.uint8))
        areas = np.bincount(labels.reshape(-1)).astype(np.int64)
        component_ids = list(range(1, num_foreground + 1))
        return labels, areas, component_ids


def make_component_filtered_mask(
        base_mask: np.ndarray,
        labels: np.ndarray | None,
        areas: np.ndarray | None,
        component_ids: List[int],
        min_component_area: int,
        keep_largest: bool,
) -> np.ndarray:
    if int(base_mask.sum()) == 0:
        return np.zeros_like(base_mask, dtype=np.uint8)
    if min_component_area <= 0 and not keep_largest:
        return base_mask.astype(np.uint8, copy=False)
    if labels is None or areas is None or not component_ids:
        return np.zeros_like(base_mask, dtype=np.uint8)

    if keep_largest:
        keep_ids = {max(component_ids, key=lambda cid: int(areas[cid]))}
    else:
        keep_ids = set(component_ids)

    if min_component_area > 0:
        keep_ids = {cid for cid in keep_ids if int(areas[cid]) >= int(min_component_area)}

    if not keep_ids:
        return np.zeros_like(base_mask, dtype=np.uint8)

    return np.isin(labels, list(keep_ids)).astype(np.uint8)


def metric_values(pred: np.ndarray, target_bool: np.ndarray, target_area: int) -> Tuple[float, float, float]:
    """Return dice, iou, pred_area for one final prediction."""
    pred_area = int(pred.sum())
    if pred_area == 0 and target_area == 0:
        return 1.0, 1.0, 0.0

    if pred_area == 0:
        # target is positive here, because negative empty-empty was handled above
        dice = EPS / (float(target_area) + EPS)
        iou = EPS / (float(target_area) + EPS)
        return float(dice), float(iou), 0.0

    if target_area == 0:
        # negative image with non-empty prediction
        dice = EPS / (float(pred_area) + EPS)
        iou = EPS / (float(pred_area) + EPS)
        return float(dice), float(iou), float(pred_area)

    tp = int(np.logical_and(pred.astype(bool), target_bool).sum())
    fp = pred_area - tp
    fn = target_area - tp
    dice = (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    return float(dice), float(iou), float(pred_area)


def init_stats(n: int) -> Dict[str, np.ndarray]:
    return {
        "fp_negative": np.zeros(n, dtype=np.int64),
        "true_negative": np.zeros(n, dtype=np.int64),
        "fn_positive": np.zeros(n, dtype=np.int64),
        "detected_positive": np.zeros(n, dtype=np.int64),
        "sum_dice_all": np.zeros(n, dtype=np.float64),
        "sum_dice_positive": np.zeros(n, dtype=np.float64),
        "sum_dice_negative": np.zeros(n, dtype=np.float64),
        "sum_iou_positive": np.zeros(n, dtype=np.float64),
        "sum_pred_area_positive": np.zeros(n, dtype=np.float64),
        "sum_pred_area_negative": np.zeros(n, dtype=np.float64),
    }


def mean_or_zero(total: float, count: int) -> float:
    return float(total / count) if count > 0 else 0.0


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    ckpt = safe_torch_load(args.checkpoint, device=device)
    config = load_config(args.config, ckpt)
    model = build_model(config, args.checkpoint, device)
    loader = build_loader(config, args)

    min_pred_areas = args.min_pred_areas if args.min_pred_areas is not None else [args.min_pred_area]
    thresholds = [float(x) for x in args.thresholds]
    min_component_areas = [int(x) for x in args.min_component_areas]
    keep_largest_options = [bool(int(x)) for x in args.keep_largest_options]
    min_pred_areas = [int(x) for x in min_pred_areas]

    configs: List[Dict] = []
    index: Dict[Tuple[int, int, int, int], int] = {}
    for ti, threshold in enumerate(thresholds):
        for min_component_area in min_component_areas:
            for keep_largest in keep_largest_options:
                for min_pred_area in min_pred_areas:
                    idx = len(configs)
                    index[(ti, min_component_area, int(keep_largest), min_pred_area)] = idx
                    configs.append(
                        {
                            "threshold": threshold,
                            "min_component_area": min_component_area,
                            "min_pred_area": min_pred_area,
                            "keep_largest_component": int(keep_largest),
                        }
                    )

    stats = init_stats(len(configs))
    num_images = 0
    num_positive = 0
    num_negative = 0

    print("=" * 100, flush=True)
    print("[Fast Sweep]", flush=True)
    print(f"num_sweep_configs : {len(configs)}", flush=True)
    print(f"thresholds        : {thresholds}", flush=True)
    print(f"min_comp_areas    : {min_component_areas}", flush=True)
    print(f"min_pred_areas    : {min_pred_areas}", flush=True)
    print(f"keep_largest      : {[int(x) for x in keep_largest_options]}", flush=True)
    print(f"output_csv        : {args.output_csv}", flush=True)
    print("=" * 100, flush=True)

    autocast_enabled = bool(args.amp and device.type == "cuda")
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
            logits = model(images)
            probs = torch.sigmoid(logits)

        probs_np = probs[:, 0].detach().cpu().numpy()
        masks_np = (masks[:, 0].detach().cpu().numpy() >= 0.5)

        for bi in range(probs_np.shape[0]):
            prob = probs_np[bi]
            target_bool = masks_np[bi]
            target_area = int(target_bool.sum())
            is_positive = target_area > 0
            num_images += 1
            if is_positive:
                num_positive += 1
            else:
                num_negative += 1

            for ti, thr in enumerate(thresholds):
                base_mask = (prob >= thr).astype(np.uint8)
                labels = areas = None
                component_ids: List[int] = []
                if int(base_mask.sum()) > 0 and (max(min_component_areas) > 0 or any(keep_largest_options)):
                    labels, areas, component_ids = connected_components(base_mask)

                processed_cache: Dict[Tuple[int, int], Tuple[np.ndarray, int, float, float]] = {}
                for min_component_area in min_component_areas:
                    for keep_largest in keep_largest_options:
                        key = (min_component_area, int(keep_largest))
                        pred0 = make_component_filtered_mask(
                            base_mask,
                            labels,
                            areas,
                            component_ids,
                            min_component_area=min_component_area,
                            keep_largest=keep_largest,
                        )
                        pred0_area = int(pred0.sum())
                        # Metric for the component-filtered mask before min_pred_area gating.
                        dice0, iou0, pred_area0 = metric_values(pred0, target_bool, target_area)
                        processed_cache[key] = (pred0, pred0_area, dice0, iou0)

                for min_component_area in min_component_areas:
                    for keep_largest in keep_largest_options:
                        pred0, pred0_area, dice0, iou0 = processed_cache[(min_component_area, int(keep_largest))]

                        for min_pred_area in min_pred_areas:
                            idx = index[(ti, min_component_area, int(keep_largest), min_pred_area)]

                            if pred0_area < min_pred_area:
                                final_pred_area = 0
                                if is_positive:
                                    dice = EPS / (float(target_area) + EPS)
                                    iou = dice
                                else:
                                    dice = 1.0
                                    iou = 1.0
                            else:
                                final_pred_area = pred0_area
                                dice = dice0
                                iou = iou0

                            stats["sum_dice_all"][idx] += dice
                            if is_positive:
                                stats["sum_dice_positive"][idx] += dice
                                stats["sum_iou_positive"][idx] += iou
                                stats["sum_pred_area_positive"][idx] += final_pred_area
                                if final_pred_area > 0:
                                    stats["detected_positive"][idx] += 1
                                else:
                                    stats["fn_positive"][idx] += 1
                            else:
                                stats["sum_dice_negative"][idx] += dice
                                stats["sum_pred_area_negative"][idx] += final_pred_area
                                if final_pred_area > 0:
                                    stats["fp_negative"][idx] += 1
                                else:
                                    stats["true_negative"][idx] += 1

        if step % max(args.progress_every, 1) == 0:
            elapsed = time.time() - t0
            print(f"[Fast Evaluate] step={step}, seen={num_images}, elapsed={elapsed/60:.1f} min", flush=True)

    rows: List[Dict] = []
    for idx, cfg in enumerate(configs):
        rows.append(
            {
                "threshold": cfg["threshold"],
                "min_component_area": cfg["min_component_area"],
                "min_pred_area": cfg["min_pred_area"],
                "keep_largest_component": cfg["keep_largest_component"],
                "num_images": num_images,
                "num_positive": num_positive,
                "num_negative": num_negative,
                "num_false_positive_negative": int(stats["fp_negative"][idx]),
                "num_true_negative": int(stats["true_negative"][idx]),
                "num_false_negative_positive": int(stats["fn_positive"][idx]),
                "num_detected_positive": int(stats["detected_positive"][idx]),
                "negative_fp_rate": float(stats["fp_negative"][idx] / max(num_negative, 1)),
                "positive_image_recall": float(stats["detected_positive"][idx] / max(num_positive, 1)),
                "mean_dice_all": mean_or_zero(float(stats["sum_dice_all"][idx]), num_images),
                "mean_dice_positive": mean_or_zero(float(stats["sum_dice_positive"][idx]), num_positive),
                "mean_dice_negative": mean_or_zero(float(stats["sum_dice_negative"][idx]), num_negative),
                "mean_iou_positive": mean_or_zero(float(stats["sum_iou_positive"][idx]), num_positive),
                "mean_pred_area_positive": mean_or_zero(float(stats["sum_pred_area_positive"][idx]), num_positive),
                "mean_pred_area_negative": mean_or_zero(float(stats["sum_pred_area_negative"][idx]), num_negative),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_for_csv = sorted(
        rows,
        key=lambda r: (
            float(r["mean_dice_all"]),
            float(r["mean_dice_positive"]),
            -float(r["negative_fp_rate"]),
        ),
        reverse=True,
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print("=" * 100, flush=True)
    print("[Top 20 by mean_dice_all]", flush=True)
    for r in rows_for_csv[:20]:
        print(
            f"thr={r['threshold']:<5} min_comp={r['min_component_area']:<4} min_pred={r['min_pred_area']:<4} "
            f"largest={r['keep_largest_component']} all={float(r['mean_dice_all']):.4f} "
            f"pos={float(r['mean_dice_positive']):.4f} iou_pos={float(r['mean_iou_positive']):.4f} "
            f"neg_fp={float(r['negative_fp_rate']):.4f} pos_img_recall={float(r['positive_image_recall']):.4f}",
            flush=True,
        )

    rows_by_pos = sorted(
        rows,
        key=lambda r: (
            float(r["mean_dice_positive"]),
            -float(r["negative_fp_rate"]),
            float(r["mean_dice_all"]),
        ),
        reverse=True,
    )
    print("=" * 100, flush=True)
    print("[Top 20 by mean_dice_positive]", flush=True)
    for r in rows_by_pos[:20]:
        print(
            f"thr={r['threshold']:<5} min_comp={r['min_component_area']:<4} min_pred={r['min_pred_area']:<4} "
            f"largest={r['keep_largest_component']} pos={float(r['mean_dice_positive']):.4f} "
            f"iou_pos={float(r['mean_iou_positive']):.4f} all={float(r['mean_dice_all']):.4f} "
            f"neg_fp={float(r['negative_fp_rate']):.4f} pos_img_recall={float(r['positive_image_recall']):.4f}",
            flush=True,
        )
    print("=" * 100, flush=True)
    print(f"Saved to: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
