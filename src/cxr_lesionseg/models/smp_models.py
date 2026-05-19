# src/cxr_lesionseg/models/smp_models.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp


class SMPModel(nn.Module):
    """
    Wrapper for segmentation_models_pytorch models.

    Supported examples:
        - unet
        - unetplusplus
        - deeplabv3plus
    """

    def __init__(
            self,
            model_name: str = "unetplusplus",
            encoder_name: str = "resnet34",
            encoder_weights: str | None = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
    ) -> None:
        super().__init__()

        model_name = model_name.lower()

        if model_name == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )

        elif model_name in ["unetplusplus", "unet++"]:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )

        elif model_name in ["deeplabv3plus", "deeplabv3+"]:
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )

        else:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                f"Supported: unet, unetplusplus, deeplabv3plus"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)