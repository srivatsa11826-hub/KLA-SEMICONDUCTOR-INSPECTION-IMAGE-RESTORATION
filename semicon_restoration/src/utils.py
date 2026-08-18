"""Reproducibility, checkpoint, tiled inference, and accelerated image I/O."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in config: {path}")
    return config


def save_yaml(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(1, self.count)


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    def log(self, values: Mapping[str, Any]) -> None:
        row = {key: values.get(key, "") for key in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # type: ignore[attr-defined]
    if hasattr(model, "module"):
        model = model.module  # type: ignore[attr-defined]
    return model


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    config: Mapping[str, Any],
    best: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": dict(config),
        "best": dict(best),
    }


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    # PyTorch 2.6 changed weights_only's default; full training checkpoints need False.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "params", "params_ema"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint does not contain a state dictionary")
    cleaned = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def forward_tiled(
    model: nn.Module,
    inputs: torch.Tensor,
    scale: int,
    tile_size: int,
    overlap: int = 16,
) -> torch.Tensor:
    """Overlap-average tiled inference. tile_size is measured in LR pixels."""
    if tile_size <= 0:
        return model(inputs)
    if tile_size <= overlap:
        raise ValueError("tile_size must be larger than overlap")
    batch, _, height, width = inputs.shape
    if height <= tile_size and width <= tile_size:
        return model(inputs)

    stride = tile_size - overlap
    y_starts = list(range(0, max(1, height - tile_size + 1), stride))
    x_starts = list(range(0, max(1, width - tile_size + 1), stride))
    y_last, x_last = max(0, height - tile_size), max(0, width - tile_size)
    if not y_starts or y_starts[-1] != y_last:
        y_starts.append(y_last)
    if not x_starts or x_starts[-1] != x_last:
        x_starts.append(x_last)

    output: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    for top in y_starts:
        for left in x_starts:
            tile = inputs[..., top : min(top + tile_size, height), left : min(left + tile_size, width)]
            restored = model(tile)
            if output is None:
                output = restored.new_zeros(batch, restored.shape[1], height * scale, width * scale)
                weight = restored.new_zeros(batch, 1, height * scale, width * scale)
            y0, x0 = top * scale, left * scale
            y1, x1 = y0 + restored.shape[-2], x0 + restored.shape[-1]
            output[..., y0:y1, x0:x1] += restored
            weight[..., y0:y1, x0:x1] += 1
    assert output is not None and weight is not None
    return output / weight.clamp_min(1)


def save_tensor_image(tensor: torch.Tensor, path: str | Path, bit_depth: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected CxHxW output, got {array.shape}")
    array = np.transpose(array, (1, 2, 0))
    if path.suffix.lower() == ".npy":
        np.save(path, array.astype(np.float32))
        return
    if path.suffix.lower() != ".png":
        raise ValueError(f"Output must be .png or .npy, got: {path}")
    if bit_depth == 8:
        image = np.rint(array * 255.0).astype(np.uint8)
    elif bit_depth == 16:
        image = np.rint(array * 65535.0).astype(np.uint16)
    else:
        raise ValueError("bit_depth must be 8 or 16")
    if image.shape[2] == 1:
        image = image[..., 0]
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("PNG writer supports one or three channels")
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


class AsyncImageWriter:
    """Bounded asynchronous disk writer used by end-to-end inference."""

    def __init__(self, workers: int = 4, max_pending: int = 32) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self.max_pending = max(1, max_pending)
        self.pending: list[Future[None]] = []

    def submit(self, tensor: torch.Tensor, path: str | Path, bit_depth: int = 8) -> None:
        # Caller supplies a CPU tensor, so worker threads never touch a CUDA context.
        self.pending.append(self.executor.submit(save_tensor_image, tensor, path, bit_depth))
        if len(self.pending) >= self.max_pending:
            self.pending.pop(0).result()

    def close(self) -> None:
        for future in self.pending:
            future.result()
        self.pending.clear()
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "AsyncImageWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, indent=2)


class WallTimer:
    def __enter__(self) -> "WallTimer":
        self.started = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed = time.perf_counter() - self.started
