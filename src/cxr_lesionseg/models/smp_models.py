# src/cxr_lesionseg/models/smp_models.py
# -*- coding: utf-8 -*-
"""
SMP model wrapper with optional decoder attention.

Drop-in replacement for:
    src/cxr_lesionseg/models/smp_models.py

For U-Net / U-Net++:
    --decoder-attention-type scse

DeepLabV3+ does not support decoder_attention_type in SMP, so attention is
rejected for that architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def normalize_decoder_attention(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"", "none", "null", "false", "0"}:
        return None
    if value not in {"scse"}:
        raise ValueError(f"Unsupported decoder_attention_type: {value}. Supported: none, scse")
    return value


class SMPModel(nn.Module):
    """
    Wrapper for segmentation_models_pytorch models.

    Supported model_name:
        - unet
        - unetplusplus / unet++
        - deeplabv3plus / deeplabv3+

    decoder_attention_type:
        - None / "none"
        - "scse" for U-Net and U-Net++ only
    """

    def __init__(
        self,
        model_name: str = "unetplusplus",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_attention_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        model_name = model_name.lower()
        decoder_attention_type = normalize_decoder_attention(decoder_attention_type)

        if model_name == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
                decoder_attention_type=decoder_attention_type,
            )

        elif model_name in {"unetplusplus", "unet++"}:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,
                decoder_attention_type=decoder_attention_type,
            )

        elif model_name in {"deeplabv3plus", "deeplabv3+"}:
            if decoder_attention_type is not None:
                raise ValueError(
                    "DeepLabV3Plus in segmentation_models_pytorch does not support "
                    "decoder_attention_type. Use --decoder-attention-type none."
                )
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
                "Supported: unet, unetplusplus, deeplabv3plus"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
