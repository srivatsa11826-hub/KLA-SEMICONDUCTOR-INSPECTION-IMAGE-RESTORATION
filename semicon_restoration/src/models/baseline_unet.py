"""Lightweight U-Net baseline with a sub-pixel super-resolution head."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )


class BaselineUNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        out_channel: int = 3,
        width: int = 32,
        scale: int = 2,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        **_: object,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("scale must be >= 1")
        channels = [width * int(m) for m in channel_multipliers]
        self.scale = int(scale)
        self.padder_size = 2 ** (len(channels) - 1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for index, current in enumerate(channels):
            # Downsampling already changes C_i to C_{i+1}; deeper blocks
            # therefore refine C_{i+1} rather than changing it again.
            input_channels = img_channel if index == 0 else current
            self.encoders.append(ConvBlock(input_channels, current))
        for index in range(len(channels) - 1):
            self.downs.append(nn.Conv2d(channels[index], channels[index + 1], 2, stride=2))

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(channels) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(channels[index], channels[index - 1], 2, stride=2))
            self.decoders.append(ConvBlock(channels[index - 1] * 2, channels[index - 1]))

        self.head = nn.Sequential(
            nn.Conv2d(channels[0], out_channel * self.scale**2, 3, padding=1),
            nn.PixelShuffle(self.scale) if self.scale > 1 else nn.Identity(),
        )
        self.skip = nn.Identity() if img_channel == out_channel else nn.Conv2d(img_channel, out_channel, 1)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        ph = (self.padder_size - height % self.padder_size) % self.padder_size
        pw = (self.padder_size - width % self.padder_size) % self.padder_size
        mode = "reflect" if height > ph and width > pw else "replicate"
        return F.pad(x, (0, pw, 0, ph), mode=mode) if ph or pw else x

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        height, width = input_tensor.shape[-2:]
        padded = self._pad(input_tensor)
        x = padded
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if index < len(self.downs):
                x = self.downs[index](x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = decoder(torch.cat([up(x), skip], dim=1))

        output = self.head(x)
        base = self.skip(padded)
        if self.scale > 1:
            base = F.interpolate(base, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return (output + base)[..., : height * self.scale, : width * self.scale]
