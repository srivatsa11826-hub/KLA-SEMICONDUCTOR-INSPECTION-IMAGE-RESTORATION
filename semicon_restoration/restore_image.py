#!/usr/bin/env python3
"""Restore one PNG or NPY image with a trained restoration checkpoint."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from src.dataset import load_image
from src.models.init import build_model
from src.utils import (
    extract_state_dict,
    forward_tiled,
    load_yaml,
    resolve_device,
    save_tensor_image,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore one degraded image")
    parser.add_argument("--input_file", required=True, help="Input .npy or .png")
    parser.add_argument("--output_file", default=None, help="Output .png or .npy")
    parser.add_argument("--weights", default="weights/real_kla/best_model.pth")
    parser.add_argument("--config", default="configs/real_kla_finetune.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bit_depth", type=int, choices=(8, 16), default=16)
    parser.add_argument("--tile_size", type=int, default=0)
    parser.add_argument("--tile_overlap", type=int, default=16)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def default_output(input_file: Path) -> Path:
    return Path("restored") / f"{input_file.stem}_restored.png"


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    output_file = Path(args.output_file) if args.output_file else default_output(input_file)
    if output_file.suffix.lower() not in {".png", ".npy"}:
        raise ValueError("output_file must end in .png or .npy")

    device = resolve_device(args.device)
    checkpoint = torch_load(args.weights, map_location="cpu")
    fallback = load_yaml(args.config)
    config: Mapping[str, Any] = (
        checkpoint["config"]
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), Mapping)
        else fallback
    )
    model_config = dict(config["model"])
    channels = int(model_config.get("img_channel", 1))
    scale = int(model_config.get("scale", 1))
    model = build_model(model_config)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable")
        model = torch.compile(model, mode="reduce-overhead")

    image = load_image(input_file, channels=channels)
    input_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    use_amp = bool(args.amp) and device.type == "cuda"
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=use_amp
    ):
        restored = forward_tiled(
            model,
            input_tensor,
            scale=scale,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
        ).clamp(0.0, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    save_tensor_image(restored[0].float().cpu(), output_file, bit_depth=args.bit_depth)

    print(f"Input:  {input_file} | shape={tuple(input_tensor.shape[1:])}")
    print(f"Output: {output_file} | shape={tuple(restored.shape[1:])}")
    print(f"Weights: {args.weights} | device={device} | time={elapsed:.3f}s")


if __name__ == "__main__":
    main()
