# src/cxr_lesionseg/datasets/siim_acr.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class SIIMACRDataset(Dataset):
    """
    SIIM-ACR Pneumothorax Segmentation Dataset.

    Output for train / val:
        {
            "image": FloatTensor [C, H, W],
            "mask":  FloatTensor [1, H, W],
            "image_id": str,
            "image_path": str,
            "mask_path": str,
        }

    Output for test:
        {
            "image": FloatTensor [C, H, W],
            "image_id": str,
            "image_path": str,
        }
    """

    def __init__(
            self,
            processed_root: Union[str, Path] = "data/processed/siim_acr_pneumothorax",
            split_root: Union[str, Path] = "data/splits/siim_acr_pneumothorax",
            split: str = "train",
            image_size: Union[int, Tuple[int, int]] = 512,
            in_channels: int = 1,
            normalize: bool = True,
            mean: Optional[Union[float, Sequence[float]]] = None,
            std: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected train / val / test.")

        if in_channels not in {1, 3}:
            raise ValueError(f"in_channels must be 1 or 3, got {in_channels}")

        self.processed_root = Path(processed_root)
        self.split_root = Path(split_root)
        self.split = split
        self.image_size = self._parse_image_size(image_size)
        self.in_channels = in_channels
        self.normalize = normalize

        if mean is None:
            mean = 0.5 if in_channels == 1 else (0.5, 0.5, 0.5)
        if std is None:
            std = 0.5 if in_channels == 1 else (0.5, 0.5, 0.5)

        self.mean = self._to_channel_array(mean, in_channels)
        self.std = self._to_channel_array(std, in_channels)

        self.image_dir = self.processed_root / ("test" if split == "test" else "train")
        self.mask_dir = self.processed_root / "masks"
        self.split_file = self.split_root / f"{split}.txt"

        self.image_ids = self._load_split_ids(self.split_file)

        if len(self.image_ids) == 0:
            raise RuntimeError(f"No ImageIds found in split file: {self.split_file}")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        if self.split != "test" and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_id = self.image_ids[index]

        image_path = self._find_image_path(self.image_dir, image_id)
        image = self._load_image(image_path)

        image = self._resize_image(image, self.image_size)
        image_tensor = self._image_to_tensor(image)

        sample: Dict[str, object] = {
            "image": image_tensor,
            "image_id": image_id,
            "image_path": str(image_path),
        }

        if self.split != "test":
            mask_path = self.mask_dir / f"{image_id}.png"
            mask = self._load_mask(mask_path)
            mask = self._resize_mask(mask, self.image_size)
            mask_tensor = self._mask_to_tensor(mask)

            sample["mask"] = mask_tensor
            sample["mask_path"] = str(mask_path)

        return sample

    @staticmethod
    def _parse_image_size(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(image_size, int):
            return image_size, image_size

        if len(image_size) != 2:
            raise ValueError(f"image_size should be int or (height, width), got {image_size}")

        height, width = int(image_size[0]), int(image_size[1])
        return height, width

    @staticmethod
    def _to_channel_array(
            value: Union[float, Sequence[float]],
            in_channels: int,
    ) -> np.ndarray:
        if isinstance(value, (int, float)):
            arr = np.array([float(value)] * in_channels, dtype=np.float32)
        else:
            arr = np.array(list(value), dtype=np.float32)

        if arr.shape[0] != in_channels:
            raise ValueError(
                f"Expected {in_channels} values, got {arr.shape[0]}: {value}"
            )

        return arr.reshape(in_channels, 1, 1)

    @staticmethod
    def _load_split_ids(split_file: Path) -> List[str]:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_file}")

        with split_file.open("r", encoding="utf-8") as f:
            image_ids = [line.strip() for line in f if line.strip()]

        return image_ids

    @staticmethod
    def _find_image_path(image_dir: Path, image_id: str) -> Path:
        for ext in IMAGE_EXTS:
            path = image_dir / f"{image_id}{ext}"
            if path.exists():
                return path

        candidates = [
            p for p in image_dir.glob(f"{image_id}.*")
            if p.suffix.lower() in IMAGE_EXTS
        ]

        if candidates:
            return candidates[0]

        raise FileNotFoundError(f"Image not found for ImageId={image_id} in {image_dir}")

    def _load_image(self, image_path: Path) -> Image.Image:
        if self.in_channels == 1:
            return Image.open(image_path).convert("L")
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _load_mask(mask_path: Path) -> Image.Image:
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

        return Image.open(mask_path).convert("L")

    @staticmethod
    def _resize_image(image: Image.Image, image_size: Tuple[int, int]) -> Image.Image:
        height, width = image_size

        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR

        return image.resize((width, height), resample=resample)

    @staticmethod
    def _resize_mask(mask: Image.Image, image_size: Tuple[int, int]) -> Image.Image:
        height, width = image_size

        try:
            resample = Image.Resampling.NEAREST
        except AttributeError:
            resample = Image.NEAREST

        return mask.resize((width, height), resample=resample)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image).astype(np.float32) / 255.0

        if self.in_channels == 1:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)

        if self.normalize:
            arr = (arr - self.mean) / self.std

        return torch.from_numpy(arr).float()

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
        arr = np.asarray(mask)
        arr = (arr > 127).astype(np.float32)
        arr = arr[None, :, :]

        return torch.from_numpy(arr).float()