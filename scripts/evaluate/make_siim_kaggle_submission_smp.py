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
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--min-component-area", type=int, default=0)
    p.add_argument("--min-pred-area", type=int, default=0)
    p.add_argument("--submission-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tta", choices=["none", "hflip"], default="none")
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg):
    model = SMPModel(
        model_name=cfg.get("model_name", "unetplusplus"),
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=int(cfg.get("in_channels", 3)),
        classes=1,
        decoder_attention_type=cfg.get("decoder_attention_type", None),
    )
    return model


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
    """
    SIIM/Kaggle RLE: top-to-bottom, then left-to-right.
    Empty mask is encoded as -1.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return "-1"

    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(x) for x in changes)


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

    pred_map = {}

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

        for i, image_id in enumerate(image_ids):
            prob = probs_np[i, 0]
            mask = (prob >= args.threshold).astype(np.uint8)

            mask = filter_components(mask, args.min_component_area)

            if args.min_pred_area > 0 and int(mask.sum()) < args.min_pred_area:
                mask[:] = 0

            if args.submission_size != mask.shape[0]:
                mask = cv2.resize(
                    mask,
                    (args.submission_size, args.submission_size),
                    interpolation=cv2.INTER_NEAREST,
                )

            pred_map[str(image_id)] = rle_encode(mask)

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

    print("[Done]", out_path)
    print("rows:", len(out))
    print("empty masks:", int((out["EncodedPixels"] == "-1").sum()))
    print("non-empty masks:", int((out["EncodedPixels"] != "-1").sum()))
    print(out.head())


if __name__ == "__main__":
    main()
