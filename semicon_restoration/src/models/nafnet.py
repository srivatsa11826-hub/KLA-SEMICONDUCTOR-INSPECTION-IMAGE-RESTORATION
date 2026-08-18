"""A compact NAFNet-derived restoration network with an xN reconstruction head."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """Activation-free block from Simple Baselines for Image Restoration."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        if dw_channels % 2 or ffn_channels % 2:
            raise ValueError("Expanded NAFBlock widths must be even for SimpleGate")

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, 3, padding=1, groups=dw_channels
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # Zero initialization makes every new block begin as an identity map.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.dropout1(self.conv3(y))
        x = residual + y * self.beta

        y = self.conv4(self.norm2(x))
        y = self.sg(y)
        y = self.dropout2(self.conv5(y))
        return x + y * self.gamma


class LowHighFrequencyFusion(nn.Module):
    """Cheap low/high-frequency decomposition entirely in LR feature space."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.low_projection = nn.Conv2d(channels, channels, 1)
        self.high_projection = nn.Conv2d(channels, channels, 1)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        fused = self.fuse(
            torch.cat([self.low_projection(low), self.high_projection(high)], dim=1)
        )
        return x + fused * self.scale


class NAFNetSR(nn.Module):
    """NAFNet encoder/decoder operating at LR, followed by PixelShuffle.

    The network intentionally emits an unconstrained training prediction.
    Evaluation and inference code clamp that prediction to [0, 1].
    """

    def __init__(
        self,
        img_channel: int = 3,
        out_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 4,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        dec_blk_nums: Sequence[int] = (2, 2, 2, 2),
        scale: int = 2,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
        dual_frequency: bool = True,
    ) -> None:
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have equal length")
        if scale < 1:
            raise ValueError("scale must be >= 1")

        self.scale = int(scale)
        self.img_channel = int(img_channel)
        self.out_channel = int(out_channel)
        self.padder_size = 2 ** len(enc_blk_nums)
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.frequency_fusion = (
            LowHighFrequencyFusion(width) if dual_frequency else nn.Identity()
        )

        make_block = lambda c: NAFBlock(
            c,
            dw_expand=dw_expand,
            ffn_expand=ffn_expand,
            drop_out_rate=drop_out_rate,
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        channels = width
        for block_count in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[make_block(channels) for _ in range(block_count)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = nn.Sequential(*[make_block(channels) for _ in range(middle_blk_num)])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for block_count in dec_blk_nums:
            # C -> 2C -> PixelShuffle -> C/2.
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*[make_block(channels) for _ in range(block_count)]))

        reconstruction_conv = nn.Conv2d(
            width, out_channel * self.scale * self.scale, 3, padding=1
        )
        # Begin as exact bicubic interpolation. This stabilizes early training
        # and lets the network learn only the restoration residual.
        nn.init.zeros_(reconstruction_conv.weight)
        nn.init.zeros_(reconstruction_conv.bias)
        self.reconstruction = nn.Sequential(
            reconstruction_conv,
            nn.PixelShuffle(self.scale) if self.scale > 1 else nn.Identity(),
        )
        self.skip_projection = (
            nn.Identity() if img_channel == out_channel else nn.Conv2d(img_channel, out_channel, 1)
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        if not pad_h and not pad_w:
            return x
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        original_h, original_w = input_tensor.shape[-2:]
        x_input = self._pad(input_tensor)
        x = self.frequency_fusion(self.intro(x_input))

        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x) + skip
            x = decoder(x)

        residual = self.reconstruction(x)
        base = self.skip_projection(x_input)
        if self.scale > 1:
            base = F.interpolate(base, scale_factor=self.scale, mode="bicubic", align_corners=False)
        output = residual + base
        return output[..., : original_h * self.scale, : original_w * self.scale]
