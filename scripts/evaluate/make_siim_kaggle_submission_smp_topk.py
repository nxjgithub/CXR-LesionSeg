#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset  # noqa: E402
from cxr_lesionseg.models import SMPModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--processed-root", required=True)
    p.add_argument("--split-root", required=True)
    p.add_argument("--sample-submission", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--min-component-area", type=int, default=0)
    p.add_argument("--min-pred-area", type=int, default=0)
    p.add_argument("--top-k", type=int, required=True)
    p.add_argument(
        "--score-method",
        choices=["sum_prob", "mean_prob", "max_prob", "pred_area"],
        default="sum_prob",
    )
    p.add_argument("--submission-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tta", choices=["none", "hflip"], default="none")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--rank-csv", default=None)
    return p.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg):
    return SMPModel(
        model_name=cfg.get("model_name", "unetplusplus"),
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=int(cfg.get("in_channels", 3)),
        classes=1,
        decoder_attention_type=cfg.get("decoder_attention_type", None),
    )


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt


def filter_components(mask, min_area):
    mask = mask.astype(np.uint8)
    if min_area <= 0 or mask.sum() == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label_id] = 1

    return out


def rle_encode(mask):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return "-1"

    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(x) for x in changes)


def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    gt_sum = gt.sum()
    pred_sum = pred.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 and pred_sum > 0:
        return 0.0
    if gt_sum > 0 and pred_sum == 0:
        return 0.0

    inter = np.logical_and(gt, pred).sum()
    return float(2.0 * inter / (gt_sum + pred_sum))


def calc_score(prob, mask, method):
    pred_area = int(mask.sum())

    if method == "max_prob":
        return float(prob.max())

    if pred_area == 0:
        return 0.0

    selected = prob[mask > 0]

    if method == "sum_prob":
        return float(selected.sum())
    if method == "mean_prob":
        return float(selected.mean())
    if method == "pred_area":
        return float(pred_area)

    raise ValueError(method)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = SIIMACRDataset(
        processed_root=args.processed_root,
        split_root=args.split_root,
        split=args.split,
        image_size=int(cfg.get("image_size", 512)),
        in_channels=int(cfg.get("in_channels", 3)),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    if "epoch" in ckpt:
        print(f"[Checkpoint] epoch={ckpt.get('epoch')} best_score={ckpt.get('best_score')}")

    records = []
    has_gt = args.split != "test"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
            logits = model(images)
            probs = torch.sigmoid(logits)

            if args.tta == "hflip":
                logits_flip = model(torch.flip(images, dims=[3]))
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, dims=[3])
                probs = 0.5 * (probs + probs_flip)

        probs_np = probs.detach().float().cpu().numpy()

        if has_gt and "mask" in batch:
            gt_np = batch["mask"].detach().cpu().numpy()
        else:
            gt_np = None

        for i, image_id in enumerate(image_ids):
            prob = probs_np[i, 0]
            mask = (prob >= args.threshold).astype(np.uint8)

            mask = filter_components(mask, args.min_component_area)

            if args.min_pred_area > 0 and int(mask.sum()) < args.min_pred_area:
                mask[:] = 0

            pred_area = int(mask.sum())
            score = calc_score(prob, mask, args.score_method)

            rec = {
                "image_id": str(image_id),
                "score": score,
                "pred_area": pred_area,
                "mask": mask,
            }

            if gt_np is not None:
                gt = (gt_np[i, 0] > 0.5).astype(np.uint8)
                rec["gt_mask"] = gt
                rec["gt_has"] = int(gt.sum() > 0)

            records.append(rec)

    eligible = [r for r in records if r["pred_area"] > 0 and r["score"] > 0]
    eligible = sorted(eligible, key=lambda x: x["score"], reverse=True)

    kept = eligible[: max(args.top_k, 0)]
    kept_ids = {r["image_id"] for r in kept}

    pred_map = {}
    rank_rows = []

    sorted_records = sorted(records, key=lambda x: x["score"], reverse=True)

    for rank, r in enumerate(sorted_records, start=1):
        image_id = r["image_id"]
        keep = image_id in kept_ids

        if keep:
            mask = r["mask"]
        else:
            mask = np.zeros_like(r["mask"], dtype=np.uint8)

        if args.submission_size != mask.shape[0]:
            submit_mask = cv2.resize(
                mask,
                (args.submission_size, args.submission_size),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            submit_mask = mask

        pred_map[image_id] = rle_encode(submit_mask)

        row = {
            "rank": rank,
            "image_id": image_id,
            "keep": int(keep),
            "score": r["score"],
            "pred_area": r["pred_area"],
        }

        if has_gt and "gt_mask" in r:
            gt = r["gt_mask"]
            d = dice_score(gt, mask)
            gt_has = bool(gt.sum() > 0)
            pred_has = bool(mask.sum() > 0)

            row.update({
                "gt_has": int(gt_has),
                "pred_has_after_topk": int(pred_has),
                "dice": d,
                "is_tp_image": int(gt_has and pred_has),
                "is_fp_image": int((not gt_has) and pred_has),
                "is_fn_image": int(gt_has and (not pred_has)),
                "is_tn_image": int((not gt_has) and (not pred_has)),
            })

        rank_rows.append(row)

    sample = pd.read_csv(args.sample_submission)
    rows = []
    missing = []

    for image_id in sample["ImageId"].astype(str).tolist():
        if image_id not in pred_map:
            missing.append(image_id)
        else:
            rows.append({"ImageId": image_id, "EncodedPixels": pred_map[image_id]})

    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} image ids, e.g. {missing[:5]}")

    out = pd.DataFrame(rows, columns=["ImageId", "EncodedPixels"])
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    rank_df = pd.DataFrame(rank_rows)
    if args.rank_csv:
        rank_path = Path(args.rank_csv)
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        rank_df.to_csv(rank_path, index=False)
        print("[Rank CSV]", rank_path)

    print("[Done]", out_path)
    print("rows:", len(out))
    print("requested top_k:", args.top_k)
    print("eligible non-empty before top-k:", len(eligible))
    print("kept non-empty after top-k:", int((out["EncodedPixels"] != "-1").sum()))

    if has_gt and "gt_has" in rank_df.columns:
        kept_df = rank_df[rank_df["keep"] == 1]
        tp = int(kept_df["is_tp_image"].sum())
        fp = int(kept_df["is_fp_image"].sum())
        fn = int(rank_df["is_fn_image"].sum())
        tn = int(rank_df["is_tn_image"].sum())

        num_pos = int(rank_df["gt_has"].sum())
        num_neg = int((rank_df["gt_has"] == 0).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(num_pos, 1)
        neg_fp_rate = fp / max(num_neg, 1)

        print("\n[VAL TOP-K IMAGE-LEVEL]")
        print("num_pos:", num_pos)
        print("num_neg:", num_neg)
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print("TN:", tn)
        print("precision:", precision)
        print("recall:", recall)
        print("negative_fp_rate:", neg_fp_rate)

        print("\n[VAL DICE]")
        print("mean_dice_all:", float(rank_df["dice"].mean()))
        print("mean_dice_positive:", float(rank_df.loc[rank_df["gt_has"] == 1, "dice"].mean()))
        print("mean_dice_negative:", float(rank_df.loc[rank_df["gt_has"] == 0, "dice"].mean()))

    print(out.head())


if __name__ == "__main__":
    main()
