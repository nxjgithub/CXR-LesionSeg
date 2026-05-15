# scripts/preprocess/check_dataset_loader.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_lesionseg.datasets import SIIMACRDataset


def check_split(split: str):
    dataset = SIIMACRDataset(
        processed_root="data/processed/siim_acr_pneumothorax",
        split_root="data/splits/siim_acr_pneumothorax",
        split=split,
        image_size=512,
        in_channels=1,
        normalize=True,
    )

    print("=" * 80)
    print(f"Split: {split}")
    print(f"Length: {len(dataset)}")

    sample = dataset[0]
    image = sample["image"]

    print("Image shape:", image.shape)
    print("Image dtype :", image.dtype)
    print("Image path  :", sample["image_path"])

    if "mask" in sample:
        mask = sample["mask"]
        print("Mask shape :", mask.shape)
        print("Mask dtype  :", mask.dtype)
        print("Mask values :", sorted(set(mask.reshape(-1).tolist())))
        print("Mask area   :", float(mask.sum()))
        print("Mask path   :", sample["mask_path"])
    else:
        print("Mask        : None")

    print("ImageId     :", sample["image_id"])


def main():
    check_split("train")
    check_split("val")
    check_split("test")


if __name__ == "__main__":
    main()
