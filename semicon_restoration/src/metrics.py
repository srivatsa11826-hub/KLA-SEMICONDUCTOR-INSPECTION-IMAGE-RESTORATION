"""GPU-friendly PSNR, SSIM, and LPIPS evaluation utilities."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import nn

try:
    from pytorch_msssim import ssim
except ImportError as error:  # pragma: no cover
    ssim = None
    _SSIM_IMPORT_ERROR = error

from .losses import LPIPSLoss


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    # Exact matches are reported as 100 dB rather than infinity for stable logging.
    return (-10.0 * torch.log10(mse.clamp_min(1e-10))).mean()


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if ssim is None:
        raise ImportError("Install pytorch-msssim to evaluate SSIM") from _SSIM_IMPORT_ERROR
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    smallest_side = min(prediction.shape[-2:])
    window_size = min(11, smallest_side if smallest_side % 2 else smallest_side - 1)
    window_size = max(1, window_size)
    return ssim(
        prediction,
        target,
        data_range=1.0,
        size_average=True,
        win_size=window_size,
    )


class RestorationMetrics(nn.Module):
    def __init__(
        self,
        compute_lpips: bool = True,
        lpips_net: str = "alex",
        lpips_random_backbone: bool = False,
        lpips_model: LPIPSLoss | None = None,
    ) -> None:
        super().__init__()
        self.lpips = (
            lpips_model or LPIPSLoss(lpips_net, lpips_random_backbone)
            if compute_lpips
            else None
        )

    @torch.no_grad()
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        lpips_value: torch.Tensor | float | None = None,
    ) -> dict[str, float]:
        prediction = prediction.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        values = {
            "psnr": float(batch_psnr(prediction, target).item()),
            "ssim": float(batch_ssim(prediction, target).item()),
        }
        if self.lpips is not None:
            # Validation can reuse the LPIPS value already produced by the
            # composite loss, avoiding a second AlexNet forward per batch.
            if lpips_value is None:
                lpips_value = self.lpips(prediction, target)
            if isinstance(lpips_value, torch.Tensor):
                lpips_value = float(lpips_value.detach().item())
            values["lpips"] = float(lpips_value)
        return values


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, values: Mapping[str, float], batch_size: int) -> None:
        self.count += int(batch_size)
        for key, value in values.items():
            self.sums[key] += float(value) * batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {key: 0.0 for key in self.sums}
        return {key: value / self.count for key, value in self.sums.items()}
