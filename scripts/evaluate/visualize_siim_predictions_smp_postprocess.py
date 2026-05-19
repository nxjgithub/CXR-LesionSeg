# scripts/evaluate/visualize_siim_predictions.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset
try:
    from cxr_lesionseg.models import UNet, SMPModel
except ImportError:
    from cxr_lesionseg.models import UNet
    from cxr_lesionseg.models.smp_models import SMPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SIIM-ACR validation predictions."
    )

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed/siim_acr_pneumothorax",
    )
    parser.add_argument(
        "--split-root",
        type=str,
        default="data/splits/siim_acr_pneumothorax",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
    )

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Prediction threshold. Use the best threshold from threshold_sweep.csv.",
    )
    parser.add_argument(
        "--min-pred-area",
        type=int,
        default=30,
        help="For negative images, predictions >= this area are treated as false positives.",
    )

    parser.add_argument(
        "--min-component-area",
        type=int,
        default=0,
        help="Remove connected components smaller than this area after thresholding.",
    )
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Keep only the largest connected component after thresholding.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of samples saved for each error category.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate first N samples. Useful for debugging.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision during inference on CUDA.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Visualization output directory. Default: checkpoint run_dir/visualizations/...",
    )

    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_config(config: Dict, device: torch.device) -> torch.nn.Module:
    in_channels = int(config.get("in_channels", 1))
    model_name = str(config.get("model_name", "")).lower()

    if model_name in ["unet", "unetplusplus", "unet++", "deeplabv3plus", "deeplabv3+"]:
        # 推理时不需要重新下载 ImageNet 权重，后面会加载 checkpoint state_dict。
        model = SMPModel(
            model_name=model_name,
            encoder_name=str(config.get("encoder_name", "resnet34")),
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
    else:
        model = UNet(
            in_channels=in_channels,
            out_channels=1,
            base_channels=int(config.get("base_channels", 32)),
        )

    return model.to(device)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    ckpt = safe_torch_load(checkpoint_path, device=device)
    config = ckpt.get("config", {})

    model = build_model_from_config(config=config, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 80)
    print("[Checkpoint]")
    print(f"path            : {checkpoint_path}")
    print(f"epoch           : {ckpt.get('epoch')}")
    print(f"best_score      : {ckpt.get('best_score', ckpt.get('best_dice'))}")
    print(f"model_name      : {config.get('model_name', 'plain_unet')}")
    print(f"encoder_name    : {config.get('encoder_name', '-')}")
    print(f"in_channels     : {config.get('in_channels', 1)}")
    print(f"image_size      : {config.get('image_size', '-')}")
    print("=" * 80)

    return model, config

def build_dataset_and_loader(args: argparse.Namespace, config: Dict) -> Tuple[SIIMACRDataset, DataLoader]:
    image_size = int(config.get("image_size", args.image_size))
    in_channels = int(config.get("in_channels", 1))

    dataset = SIIMACRDataset(
        processed_root=args.processed_root,
        split_root=args.split_root,
        split=args.split,
        image_size=image_size,
        in_channels=in_channels,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=False,
    )

    print("=" * 80)
    print("[Dataset]")
    print(f"split        : {args.split}")
    print(f"length       : {len(dataset)}")
    print(f"image_size   : {image_size}")
    print(f"in_channels  : {in_channels}")
    print(f"batch_size   : {args.batch_size}")
    print("=" * 80)

    return dataset, loader

def get_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)

    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent.parent

    threshold_tag = f"{args.threshold:.2f}".replace(".", "p")
    return run_dir / "visualizations" / f"{args.split}_predictions_thr{threshold_tag}"


def compute_binary_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-7,
) -> Dict[str, float]:
    """
    pred, target: [1, H, W], float tensor, values 0/1.
    """

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1.0 - target_flat)).sum()
    fn = ((1.0 - pred_flat) * target_flat).sum()
    tn = ((1.0 - pred_flat) * (1.0 - target_flat)).sum()

    pred_area = pred_flat.sum()
    target_area = target_flat.sum()

    if float(pred_area + target_area) == 0.0:
        dice = torch.tensor(1.0)
        iou = torch.tensor(1.0)
    else:
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "specificity": float(specificity.item()),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "fn": float(fn.item()),
        "tn": float(tn.item()),
        "pred_area": int(pred_area.item()),
        "target_area": int(target_area.item()),
    }



def postprocess_mask(
        mask: np.ndarray,
        min_component_area: int = 0,
        keep_largest_component: bool = False,
) -> np.ndarray:
    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        return mask

    if min_component_area <= 0 and not keep_largest_component:
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

    if keep_largest_component:
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

    return np.isin(labels, list(keep_ids)).astype(np.uint8)

@torch.no_grad()
def evaluate_per_image(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        threshold: float,
        min_pred_area: int,
        min_component_area: int = 0,
        keep_largest_component: bool = False,
        max_samples: int | None = None,
        use_amp: bool = False,
) -> List[Dict]:
    records: List[Dict] = []
    seen = 0

    autocast_enabled = bool(use_amp and device.type == "cuda")

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(
                device_type="cuda",
                enabled=autocast_enabled,
        ):
            logits = model(images)
            probs = torch.sigmoid(logits)

        batch_size = images.size(0)

        for i in range(batch_size):
            global_index = seen + i

            if max_samples is not None and global_index >= max_samples:
                return records

            mask_i = masks[i].detach().cpu()
            prob_i = probs[i].detach().cpu()

            pred_np = (prob_i.squeeze(0).numpy() >= threshold).astype(np.uint8)
            pred_np = postprocess_mask(
                pred_np,
                min_component_area=min_component_area,
                keep_largest_component=keep_largest_component,
            )
            pred_i = torch.from_numpy(pred_np).unsqueeze(0).float()

            metrics = compute_binary_metrics(pred_i, mask_i)

            target_area = metrics["target_area"]
            pred_area = metrics["pred_area"]

            if target_area > 0:
                image_type = "positive"
                is_false_positive_negative = False
                is_true_negative = False
            else:
                image_type = "negative"
                is_false_positive_negative = pred_area >= min_pred_area
                is_true_negative = pred_area < min_pred_area

            record = {
                "index": global_index,
                "image_id": batch["image_id"][i],
                "image_path": batch["image_path"][i],
                "mask_path": batch["mask_path"][i],
                "image_type": image_type,
                "is_false_positive_negative": int(is_false_positive_negative),
                "is_true_negative": int(is_true_negative),
                "threshold": threshold,
                "min_pred_area": min_pred_area,
                "prob_max": float(prob_i.max().item()),
                "prob_mean": float(prob_i.mean().item()),
                **metrics,
            }

            records.append(record)

        seen += batch_size

        if step % 20 == 0:
            print(f"[Evaluate] step={step}, seen={seen}")

    return records


def save_records_csv(records: List[Dict], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(records) == 0:
        return

    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def summarize_records(records: List[Dict]) -> Dict:
    if len(records) == 0:
        return {}

    positive = [r for r in records if r["target_area"] > 0]
    negative = [r for r in records if r["target_area"] == 0]
    fp_neg = [r for r in negative if r["pred_area"] >= r["min_pred_area"]]
    tn_neg = [r for r in negative if r["pred_area"] < r["min_pred_area"]]

    def mean_value(rows: List[Dict], key: str) -> float:
        if len(rows) == 0:
            return 0.0
        return float(np.mean([float(r[key]) for r in rows]))

    summary = {
        "num_images": len(records),
        "num_positive": len(positive),
        "num_negative": len(negative),
        "num_false_positive_negative": len(fp_neg),
        "num_true_negative": len(tn_neg),
        "negative_fp_rate": len(fp_neg) / max(len(negative), 1),
        "mean_dice_all": mean_value(records, "dice"),
        "mean_dice_positive": mean_value(positive, "dice"),
        "mean_dice_negative": mean_value(negative, "dice"),
        "mean_iou_positive": mean_value(positive, "iou"),
        "mean_pred_area_positive": mean_value(positive, "pred_area"),
        "mean_pred_area_negative": mean_value(negative, "pred_area"),
    }

    return summary


def save_summary_json(summary: Dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def make_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.45,
) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)

    color_arr = np.zeros_like(overlay)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]

    overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - alpha)
            + color_arr[mask_bool] * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def load_original_image(image_path: str, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(image)


def save_prediction_figure(
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        prob_map: np.ndarray,
        record: Dict,
        category: str,
        save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt_overlay = make_overlay(
        image=image,
        mask=gt_mask,
        color=(255, 0, 0),
        alpha=0.45,
    )

    pred_overlay = make_overlay(
        image=image,
        mask=pred_mask,
        color=(0, 255, 0),
        alpha=0.45,
    )

    both_overlay = image.copy().astype(np.float32)

    gt_bool = gt_mask.astype(bool)
    pred_bool = pred_mask.astype(bool)

    red = np.zeros_like(both_overlay)
    red[..., 0] = 255

    green = np.zeros_like(both_overlay)
    green[..., 1] = 255

    both_overlay[gt_bool] = both_overlay[gt_bool] * 0.55 + red[gt_bool] * 0.45
    both_overlay[pred_bool] = both_overlay[pred_bool] * 0.55 + green[pred_bool] * 0.45
    both_overlay = np.clip(both_overlay, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_mask, cmap="gray")
    ax2.set_title(f"GT mask | area={record['target_area']}")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(pred_mask, cmap="gray")
    ax3.set_title(f"Pred mask | area={record['pred_area']}")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(gt_overlay)
    ax4.set_title("GT overlay | red")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(pred_overlay)
    ax5.set_title("Pred overlay | green")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(prob_map, cmap="gray", vmin=0.0, vmax=1.0)
    ax6.set_title(f"Probability map | max={record['prob_max']:.3f}")
    ax6.axis("off")

    title = (
        f"{category} | {record['image_id']}\n"
        f"dice={record['dice']:.4f}, iou={record['iou']:.4f}, "
        f"precision={record['precision']:.4f}, recall={record['recall']:.4f}, "
        f"threshold={record['threshold']:.2f}"
    )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # 额外保存一张 GT + Pred 叠加图，方便论文或报告里快速挑图。
    compare_path = save_path.with_name(save_path.stem + "_compare.png")
    Image.fromarray(both_overlay).save(compare_path)


@torch.no_grad()
def visualize_selected_records(
        model: torch.nn.Module,
        dataset: SIIMACRDataset,
        records: List[Dict],
        output_dir: Path,
        category: str,
        device: torch.device,
        threshold: float,
        image_size: int,
        min_component_area: int,
        keep_largest_component: bool,
        use_amp: bool,
) -> None:
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    autocast_enabled = bool(use_amp and device.type == "cuda")

    for rank, record in enumerate(records, start=1):
        index = int(record["index"])
        sample = dataset[index]

        image_tensor = sample["image"].unsqueeze(0).to(device)
        gt_mask_tensor = sample["mask"]

        with torch.amp.autocast(
                device_type="cuda",
                enabled=autocast_enabled,
        ):
            logits = model(image_tensor)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        pred_mask = (prob >= threshold).astype(np.uint8)


        pred_mask = postprocess_mask(


            pred_mask,


            min_component_area=min_component_area,


            keep_largest_component=keep_largest_component,


        )
        pred_mask = postprocess_mask(
            pred_mask,
            min_component_area=min_component_area,
            keep_largest_component=keep_largest_component,
        )
        gt_mask = gt_mask_tensor[0].detach().cpu().numpy().astype(np.uint8)

        image = load_original_image(
            image_path=sample["image_path"],
            image_size=image_size,
        )

        image_id = str(record["image_id"])
        save_name = (
            f"{rank:03d}_dice{record['dice']:.4f}_"
            f"pred{record['pred_area']}_gt{record['target_area']}_"
            f"{image_id}.png"
        )

        save_path = category_dir / save_name

        save_prediction_figure(
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            prob_map=prob,
            record=record,
            category=category,
            save_path=save_path,
        )

    print(f"[Saved] {category}: {category_dir}")


def select_records(records: List[Dict], top_k: int, min_pred_area: int) -> Dict[str, List[Dict]]:
    positive = [r for r in records if r["target_area"] > 0]
    negative = [r for r in records if r["target_area"] == 0]

    high_dice_positive = sorted(
        positive,
        key=lambda x: x["dice"],
        reverse=True,
    )[:top_k]

    low_dice_positive = sorted(
        positive,
        key=lambda x: x["dice"],
        reverse=False,
    )[:top_k]

    false_positive_negative = sorted(
        [r for r in negative if r["pred_area"] >= min_pred_area],
        key=lambda x: x["pred_area"],
        reverse=True,
    )[:top_k]

    true_negative = sorted(
        [r for r in negative if r["pred_area"] < min_pred_area],
        key=lambda x: x["pred_area"],
        reverse=False,
    )[:top_k]

    return {
        "high_dice_positive": high_dice_positive,
        "low_dice_positive": low_dice_positive,
        "false_positive_negative": false_positive_negative,
        "true_negative": true_negative,
    }


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    output_dir = get_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[Visualize SIIM Predictions]")
    print(f"device        : {device}")
    print(f"checkpoint    : {args.checkpoint}")
    print(f"threshold     : {args.threshold}")
    print(f"min_pred_area : {args.min_pred_area}")
    print(f"min_comp_area : {args.min_component_area}")
    print(f"keep_largest  : {args.keep_largest_component}")
    print(f"output_dir    : {output_dir}")
    print("=" * 80)

    model, config = load_model(args.checkpoint, device=device)
    dataset, loader = build_dataset_and_loader(args, config)

    records = evaluate_per_image(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
        min_pred_area=args.min_pred_area,
        min_component_area=args.min_component_area,
        keep_largest_component=args.keep_largest_component,
        max_samples=args.max_samples,
        use_amp=args.amp,
    )

    records_csv_path = output_dir / "per_image_metrics.csv"
    save_records_csv(records, records_csv_path)

    summary = summarize_records(records)
    summary_path = output_dir / "summary.json"
    save_summary_json(summary, summary_path)

    print("=" * 80)
    print("[Summary]")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 80)

    selected = select_records(
        records=records,
        top_k=args.top_k,
        min_pred_area=args.min_pred_area,
    )

    selected_summary = {
        category: [
            {
                "index": r["index"],
                "image_id": r["image_id"],
                "dice": r["dice"],
                "iou": r["iou"],
                "pred_area": r["pred_area"],
                "target_area": r["target_area"],
                "prob_max": r["prob_max"],
            }
            for r in rows
        ]
        for category, rows in selected.items()
    }

    selected_json_path = output_dir / "selected_cases.json"
    save_summary_json(selected_summary, selected_json_path)

    for category, rows in selected.items():
        visualize_selected_records(
            model=model,
            dataset=dataset,
            records=rows,
            output_dir=output_dir,
            category=category,
            device=device,
            threshold=args.threshold,
            image_size=int(config.get("image_size", args.image_size)),
            use_amp=args.amp,
        )

    print("=" * 80)
    print("[Done]")
    print(f"Per-image metrics : {records_csv_path}")
    print(f"Summary           : {summary_path}")
    print(f"Selected cases    : {selected_json_path}")
    print(f"Visualizations    : {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()