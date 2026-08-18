"""Model registry. The filename follows the requested repository layout."""
from __future__ import annotations

from typing import Any, Mapping

from .baseline_unet import BaselineUNet
from .nafnet import NAFNetSR


def build_model(config: Mapping[str, Any]):
    config = dict(config)
    name = str(config.pop("name", "nafnet")).lower()
    if name in {"nafnet", "nafnet_sr", "nafnet-kla"}:
        return NAFNetSR(**config)
    if name in {"unet", "baseline_unet"}:
        return BaselineUNet(**config)
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["NAFNetSR", "BaselineUNet", "build_model"]
