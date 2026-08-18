"""Paired geometry and synthetic KLA-style degradation operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def _range_pair(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return 0.0, float(value)
    if len(value) != 2:
        raise ValueError(f"Expected [minimum, maximum], got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if lo < 0 or hi < lo:
        raise ValueError(f"Invalid sigma range: {value!r}")
    return lo, hi


def _ensure_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")
    return image


@dataclass
class RandomPairedGeometry:
    """Apply exactly synchronized lossless transforms to an image pair."""

    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5

    def __call__(
        self,
        noisy: np.ndarray,
        gt: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < self.horizontal_flip_p:
            noisy, gt = noisy[:, ::-1], gt[:, ::-1]
        if rng.random() < self.vertical_flip_p:
            noisy, gt = noisy[::-1, :], gt[::-1, :]
        if rng.random() < self.rotate90_p:
            k = int(rng.integers(1, 4))
            noisy, gt = np.rot90(noisy, k), np.rot90(gt, k)
        return np.ascontiguousarray(noisy), np.ascontiguousarray(gt)


class SyntheticDegradation:
    """Apply Gaussian, speckle, and downsampling in a random permutation.

    Floating-point outputs are intentionally *not* clipped. Consequently,
    noise may produce values outside [0, 1], matching the challenge input.
    """

    def __init__(
        self,
        scale: int | Sequence[int] = 2,
        gaussian_sigma: float | Sequence[float] = (0.0, 0.08),
        speckle_sigma: float | Sequence[float] = (0.0, 0.12),
        interpolation: str | Sequence[str] = ("bicubic", "bilinear"),
        random_order: bool = True,
    ) -> None:
        scales = [int(scale)] if isinstance(scale, (int, float)) else [int(x) for x in scale]
        if not scales or any(x < 1 for x in scales):
            raise ValueError("scale must contain positive integers")
        self.scales = tuple(scales)
        self.gaussian_sigma = _range_pair(gaussian_sigma)
        self.speckle_sigma = _range_pair(speckle_sigma)
        interpolations = [interpolation] if isinstance(interpolation, str) else list(interpolation)
        unknown = set(interpolations) - set(_INTERPOLATIONS)
        if unknown:
            raise ValueError(f"Unknown interpolation(s): {sorted(unknown)}")
        self.interpolations = tuple(interpolations)
        self.random_order = bool(random_order)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SyntheticDegradation":
        return cls(
            scale=config.get("scale", 2),
            gaussian_sigma=config.get("gaussian_sigma", (0.0, 0.08)),
            speckle_sigma=config.get("speckle_sigma", (0.0, 0.12)),
            interpolation=config.get("interpolation", ("bicubic", "bilinear")),
            random_order=config.get("random_order", True),
        )

    def sample_scale(self, rng: np.random.Generator) -> int:
        return int(self.scales[int(rng.integers(0, len(self.scales)))])

    def __call__(
        self,
        image: np.ndarray,
        rng: np.random.Generator | None = None,
        scale: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rng = rng or np.random.default_rng()
        output = _ensure_hwc(np.asarray(image, dtype=np.float32)).copy()
        selected_scale = int(scale if scale is not None else self.sample_scale(rng))
        sigma_g = float(rng.uniform(*self.gaussian_sigma))
        sigma_s = float(rng.uniform(*self.speckle_sigma))
        interpolation = self.interpolations[int(rng.integers(0, len(self.interpolations)))]

        operations = ["gaussian", "speckle", "downsample"]
        if self.random_order:
            operations = [operations[i] for i in rng.permutation(len(operations))]

        for operation in operations:
            if operation == "gaussian" and sigma_g > 0:
                output += rng.normal(0.0, sigma_g, output.shape).astype(np.float32)
            elif operation == "speckle" and sigma_s > 0:
                noise = rng.normal(0.0, sigma_s, output.shape).astype(np.float32)
                output += output * noise
            elif operation == "downsample" and selected_scale > 1:
                height, width = output.shape[:2]
                target = (max(1, width // selected_scale), max(1, height // selected_scale))
                output = cv2.resize(output, target, interpolation=_INTERPOLATIONS[interpolation])
                output = _ensure_hwc(output)

        metadata = {
            "scale": selected_scale,
            "gaussian_sigma": sigma_g,
            "speckle_sigma": sigma_s,
            "interpolation": interpolation,
            "order": operations,
        }
        return np.ascontiguousarray(output, dtype=np.float32), metadata
