"""Datasets for paired and synthetic semiconductor image restoration."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import RandomPairedGeometry, SyntheticDegradation

SUPPORTED_SUFFIXES = {".png", ".npy"}


def _scan_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No .png or .npy images found under: {root}")
    return files


def _to_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {array.shape}")
    # NPY files are often stored as CxHxW. Avoid transposing tiny HxWxC images.
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    return array


def load_image(path: str | Path, channels: int = 3) -> np.ndarray:
    """Load PNG/NPY as float32 HWC while preserving floating input range."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".png":
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise OSError(f"OpenCV failed to read image: {path}")
        if array.ndim == 3:
            if array.shape[2] == 4:
                array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)[..., :3]
            else:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    original_dtype = array.dtype
    array = _to_hwc(np.asarray(array))
    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        array = array.astype(np.float32) / max_value
    else:
        # Crucially, floating NPY data is not range-clipped or auto-rescaled.
        array = array.astype(np.float32, copy=False)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)

    if array.shape[2] == 4:
        array = array[..., :3]
    if channels == 1 and array.shape[2] != 1:
        # ITU-R BT.601 luminance; input is RGB at this point.
        array = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )[..., None]
    elif channels == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] != channels:
        raise ValueError(f"{path} has {array.shape[2]} channels; requested {channels}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def _pad_bottom_right(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    pad_h = max(0, target_h - image.shape[0])
    pad_w = max(0, target_w - image.shape[1])
    if not pad_h and not pad_w:
        return image
    mode = "reflect" if image.shape[0] > 1 and image.shape[1] > 1 else "edge"
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)


def _crop_gt(
    gt: np.ndarray,
    patch_size: int,
    scale: int,
    rng: np.random.Generator,
    random_crop: bool,
) -> np.ndarray:
    patch_size = (int(patch_size) // scale) * scale
    if patch_size < scale:
        raise ValueError(f"gt_patch_size must be at least scale={scale}")
    gt = _pad_bottom_right(gt, patch_size, patch_size)
    max_top, max_left = gt.shape[0] - patch_size, gt.shape[1] - patch_size
    if random_crop:
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_left + 1))
    else:
        top, left = max_top // 2, max_left // 2
    return gt[top : top + patch_size, left : left + patch_size]


def _crop_aligned_pair(
    noisy: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    scale: int,
    rng: np.random.Generator,
    random_crop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gt_patch = (int(patch_size) // scale) * scale
    if gt_patch < scale:
        raise ValueError(f"gt_patch_size must be at least scale={scale}")
    lr_patch = gt_patch // scale
    noisy = _pad_bottom_right(noisy, lr_patch, lr_patch)
    gt = _pad_bottom_right(gt, lr_patch * scale, lr_patch * scale)
    max_top, max_left = noisy.shape[0] - lr_patch, noisy.shape[1] - lr_patch
    if random_crop:
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_left + 1))
    else:
        top, left = max_top // 2, max_left // 2
    noisy_crop = noisy[top : top + lr_patch, left : left + lr_patch]
    gt_top, gt_left = top * scale, left * scale
    gt_crop = gt[gt_top : gt_top + gt_patch, gt_left : gt_left + gt_patch]
    return noisy_crop, gt_crop


class RestorationDataset(Dataset[dict[str, Any]]):
    """Paired NoisyLR/GT dataset, or on-the-fly synthetic dataset.

    Pairing is strict and based on each file's relative path without suffix.
    This catches silent image/target misalignment before training begins.
    """

    def __init__(
        self,
        gt_dir: str | Path,
        noisy_dir: str | Path | None = None,
        channels: int = 3,
        gt_patch_size: int | None = 256,
        training: bool = True,
        augment: bool = True,
        synthetic_degradation: SyntheticDegradation | None = None,
        expected_scale: int | None = None,
        seed: int = 42,
        deterministic_synthetic: bool = False,
    ) -> None:
        self.gt_root = Path(gt_dir)
        self.noisy_root = Path(noisy_dir) if noisy_dir else None
        self.channels = int(channels)
        self.gt_patch_size = int(gt_patch_size) if gt_patch_size else None
        self.training = bool(training)
        self.geometry = RandomPairedGeometry() if augment else None
        self.synthetic = synthetic_degradation
        self.expected_scale = int(expected_scale) if expected_scale else None
        self.seed = int(seed)
        self.deterministic_synthetic = bool(deterministic_synthetic)

        gt_files = _scan_images(self.gt_root)
        if self.noisy_root is None:
            if self.synthetic is None:
                raise ValueError("synthetic_degradation is required when noisy_dir is null")
            if self.expected_scale is not None and any(
                scale != self.expected_scale for scale in self.synthetic.scales
            ):
                raise ValueError(
                    f"Synthetic scales {self.synthetic.scales} must all equal the fixed "
                    f"model scale {self.expected_scale} for batched super-resolution"
                )
            self.samples = [(None, gt_path, gt_path.relative_to(self.gt_root).with_suffix("").as_posix()) for gt_path in gt_files]
        else:
            noisy_files = _scan_images(self.noisy_root)
            gt_map = {p.relative_to(self.gt_root).with_suffix("").as_posix(): p for p in gt_files}
            noisy_map = {p.relative_to(self.noisy_root).with_suffix("").as_posix(): p for p in noisy_files}
            missing_noisy = sorted(set(gt_map) - set(noisy_map))
            missing_gt = sorted(set(noisy_map) - set(gt_map))
            if missing_noisy or missing_gt:
                message = ["NoisyLR/GT pairing mismatch."]
                if missing_noisy:
                    message.append(f"Missing NoisyLR ({len(missing_noisy)}): {missing_noisy[:5]}")
                if missing_gt:
                    message.append(f"Missing GT ({len(missing_gt)}): {missing_gt[:5]}")
                raise ValueError(" ".join(message))
            self.samples = [(noisy_map[key], gt_map[key], key) for key in sorted(gt_map)]

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic_synthetic or not self.training:
            return np.random.default_rng(self.seed + index * 104729)
        # np.random is initialized per worker by seed_worker in utils.py.
        return np.random.default_rng(int(np.random.randint(0, 2**32 - 1)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        noisy_path, gt_path, key = self.samples[index]
        rng = self._rng(index)
        gt = np.clip(load_image(gt_path, self.channels), 0.0, 1.0)
        metadata: dict[str, Any] = {}

        if noisy_path is None:
            assert self.synthetic is not None
            scale = self.synthetic.sample_scale(rng)
            if self.expected_scale is not None and scale != self.expected_scale:
                raise ValueError(f"Synthetic scale {scale} != model scale {self.expected_scale}")
            if self.gt_patch_size:
                gt = _crop_gt(gt, self.gt_patch_size, scale, rng, self.training)
            else:
                # PixelShuffle produces exactly LR*scale. Pad uncropped GT-only
                # images to a scale-divisible size so prediction and target can
                # never differ by one border pixel.
                target_h = ((gt.shape[0] + scale - 1) // scale) * scale
                target_w = ((gt.shape[1] + scale - 1) // scale) * scale
                gt = _pad_bottom_right(gt, target_h, target_w)
            if self.geometry is not None:
                gt, _ = self.geometry(gt, gt, rng)
            noisy, metadata = self.synthetic(gt, rng=rng, scale=scale)
        else:
            noisy = load_image(noisy_path, self.channels)  # deliberately not clipped
            h, w = noisy.shape[:2]
            gt_h, gt_w = gt.shape[:2]
            if gt_h % h or gt_w % w or gt_h // h != gt_w // w:
                raise ValueError(
                    f"Unaligned pair {key}: NoisyLR={noisy.shape}, GT={gt.shape}; "
                    "GT dimensions must be an equal integer multiple"
                )
            scale = gt_h // h
            if self.expected_scale is not None and scale != self.expected_scale:
                raise ValueError(f"Pair {key} has scale {scale}, model expects {self.expected_scale}")
            if self.gt_patch_size:
                noisy, gt = _crop_aligned_pair(
                    noisy, gt, self.gt_patch_size, scale, rng, self.training
                )
            if self.geometry is not None:
                noisy, gt = self.geometry(noisy, gt, rng)
            metadata = {"scale": scale, "source": "paired"}

        return {
            "input": _as_tensor(noisy),
            "target": _as_tensor(gt),
            "name": key,
            "scale": int(metadata["scale"]),
        }


class InferenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, input_dir: str | Path, channels: int = 3) -> None:
        self.root = Path(input_dir)
        self.files = _scan_images(self.root)
        self.channels = int(channels)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        image = load_image(path, self.channels)
        return {
            "input": _as_tensor(image),
            "relative_path": path.relative_to(self.root).as_posix(),
        }


def list_collate(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep variable-sized inference samples as a list for shape grouping."""
    return list(batch)
