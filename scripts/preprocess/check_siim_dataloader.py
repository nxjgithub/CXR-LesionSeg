# scripts/preprocess/check_siim_dataloader.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset


def check_split(split: str, batch_size: int = 4) -> None:
    dataset = SIIMACRDataset(
        processed_root="data/processed/siim_acr_pneumothorax",
        split_root="data/splits/siim_acr_pneumothorax",
        split=split,
        image_size=512,
        in_channels=1,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    print("=" * 80)
    print(f"Split: {split}")
    print(f"Dataset length: {len(dataset)}")

    images = batch["image"]
    print("image shape:", tuple(images.shape))
    print("image dtype:", images.dtype)
    print("image min  :", float(images.min()))
    print("image max  :", float(images.max()))

    if "mask" in batch:
        masks = batch["mask"]
        print("mask shape :", tuple(masks.shape))
        print("mask dtype :", masks.dtype)
        print("mask min   :", float(masks.min()))
        print("mask max   :", float(masks.max()))
        print("mask areas :", masks.flatten(1).sum(dim=1).tolist())
        print("mask values:", torch.unique(masks).tolist())
    else:
        print("mask       : None")

    print("image_ids  :", batch["image_id"])


def main() -> None:
    check_split("train")
    check_split("val")
    check_split("test")


if __name__ == "__main__":
    main()