"""Composite Charbonnier + MS-SSIM + LPIPS restoration objective."""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import torch
from torch import nn

try:
    from pytorch_msssim import ms_ssim
except ImportError as error:  # pragma: no cover - dependency error is explained at runtime
    ms_ssim = None
    _MSSSIM_IMPORT_ERROR = error


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps_squared = float(eps) ** 2

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.eps_squared).mean()


class MultiScaleSSIMLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if ms_ssim is None:
            raise ImportError("Install pytorch-msssim to use MS-SSIM") from _MSSSIM_IMPORT_ERROR
        smallest_side = min(prediction.shape[-2:])
        # pytorch-msssim requires side > (win_size - 1) * 16 for five levels.
        max_window = max(1, (smallest_side - 1) // 16 + 1)
        window_size = min(11, max_window)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            warnings.warn("Image is too small for five-level MS-SSIM; SSIM term is disabled")
            return prediction.new_zeros(())
        similarity = ms_ssim(
            prediction,
            target,
            data_range=1.0,
            size_average=True,
            win_size=window_size,
        )
        return 1.0 - similarity


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "alex", random_backbone: bool = False) -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as error:  # pragma: no cover
            raise ImportError("Install lpips to enable the LPIPS loss/metric") from error
        try:
            self.network = lpips.LPIPS(
                net=net,
                pnet_rand=bool(random_backbone),
                verbose=False,
            )
        except Exception as error:
            raise RuntimeError(
                "LPIPS initialization failed. Its pretrained backbone may need a one-time "
                "download. Provide network access, pre-populate the torch cache, or set "
                "loss.lpips_random_backbone=true for an offline smoke test."
            ) from error
        self.network.requires_grad_(False)
        self.network.eval()

    @staticmethod
    def _three_channels(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] == 1:
            return tensor.repeat(1, 3, 1, 1)
        if tensor.shape[1] != 3:
            raise ValueError("LPIPS supports one-channel (repeated) or three-channel images")
        return tensor

    def train(self, mode: bool = True):
        super().train(mode)
        self.network.eval()
        return self

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self._three_channels(prediction) * 2.0 - 1.0
        target = self._three_channels(target) * 2.0 - 1.0
        return self.network(prediction, target, normalize=False).mean()


class CompositeRestorationLoss(nn.Module):
    def __init__(
        self,
        lambda_charb: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_lpips: float = 0.1,
        charbonnier_eps: float = 1e-3,
        lpips_net: str = "alex",
        lpips_random_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_charb = float(lambda_charb)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_lpips = float(lambda_lpips)
        self.charbonnier = CharbonnierLoss(charbonnier_eps)
        self.ssim = MultiScaleSSIMLoss()
        self.lpips = (
            LPIPSLoss(lpips_net, lpips_random_backbone) if self.lambda_lpips > 0 else None
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CompositeRestorationLoss":
        return cls(**dict(config))

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Raw output drives the pixel loss; bounded views keep perceptual losses valid.
        charb = self.charbonnier(prediction, target)
        bounded_prediction = prediction.clamp(0.0, 1.0)
        bounded_target = target.clamp(0.0, 1.0)
        ssim_loss = (
            self.ssim(bounded_prediction, bounded_target)
            if self.lambda_ssim > 0
            else prediction.new_zeros(())
        )
        lpips_loss = (
            self.lpips(bounded_prediction, bounded_target)
            if self.lambda_lpips > 0 and self.lpips is not None
            else prediction.new_zeros(())
        )
        total = (
            self.lambda_charb * charb
            + self.lambda_ssim * ssim_loss
            + self.lambda_lpips * lpips_loss
        )
        return total, {
            "total": total,
            "charbonnier": charb,
            "ms_ssim_loss": ssim_loss,
            "lpips_loss": lpips_loss,
        }
