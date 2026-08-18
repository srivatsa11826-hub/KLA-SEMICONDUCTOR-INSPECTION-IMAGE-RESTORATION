#!/usr/bin/env python3
"""High-throughput standalone restoration inference CLI."""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import InferenceDataset, list_collate
from src.models.init import build_model
from src.utils import (
    AsyncImageWriter,
    count_parameters,
    extract_state_dict,
    forward_tiled,
    load_yaml,
    resolve_device,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore KLA NoisyLR images")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--io_workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile_size", type=int, default=0, help="LR tile size; 0 disables tiling")
    parser.add_argument("--tile_overlap", type=int, default=16)
    parser.add_argument("--save_format", choices=("png", "npy", "auto"), default="png")
    parser.add_argument("--bit_depth", type=int, choices=(8, 16), default=8)
    return parser.parse_args()


def output_path(relative_path: str, output_root: Path, save_format: str) -> Path:
    relative = Path(relative_path)
    if save_format == "auto":
        suffix = ".npy" if relative.suffix.lower() == ".npy" else ".png"
    else:
        suffix = "." + save_format
    return output_root / relative.with_suffix(suffix)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch_load(args.weights, map_location="cpu")
    fallback_config = load_yaml(args.config)
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("config"), Mapping):
        config: Mapping[str, Any] = checkpoint["config"]
        print("Using architecture configuration embedded in checkpoint.")
    else:
        config = fallback_config
        print("Checkpoint has no embedded config; using --config architecture.")
    model_config = dict(config["model"])
    scale = int(model_config.get("scale", 1))
    channels = int(model_config.get("img_channel", config.get("data", {}).get("channels", 3)))

    model = build_model(model_config)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()
    use_channels_last = bool(args.channels_last) and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        model = torch.compile(model, mode="reduce-overhead")

    dataset = InferenceDataset(args.input_dir, channels=channels)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": list_collate,
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    use_amp = bool(args.amp) and device.type == "cuda"

    cuda_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cpu_model_seconds = 0.0
    image_count = 0
    started = time.perf_counter()  # Includes reads, H2D, model, D2H, clipping, and writes.
    with torch.inference_mode(), AsyncImageWriter(args.io_workers) as writer:
        for samples in tqdm(loader, desc="restore", dynamic_ncols=True):
            groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                groups[tuple(sample["input"].shape)].append(sample)

            for group in groups.values():
                inputs = torch.stack([sample["input"] for sample in group])
                inputs = inputs.to(device, non_blocking=True)
                if use_channels_last:
                    inputs = inputs.contiguous(memory_format=torch.channels_last)

                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    model_started = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = forward_tiled(
                        model, inputs, scale, args.tile_size, args.tile_overlap
                    )
                outputs = outputs.clamp(0.0, 1.0)
                if device.type == "cuda":
                    end_event.record()
                    cuda_event_pairs.append((start_event, end_event))
                else:
                    cpu_model_seconds += time.perf_counter() - model_started

                # D2H is deliberately inside the end-to-end timer.
                outputs_cpu = outputs.float().cpu()
                for restored, sample in zip(outputs_cpu, group):
                    destination = output_path(
                        sample["relative_path"], output_root, args.save_format
                    )
                    writer.submit(restored, destination, args.bit_depth)
                    image_count += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        model_seconds = sum(start.elapsed_time(end) for start, end in cuda_event_pairs) / 1000.0
    else:
        model_seconds = cpu_model_seconds
    total_seconds = time.perf_counter() - started
    print(
        f"Restored {image_count} images to {output_root}\n"
        f"Model: {count_parameters(model):,} parameters | scale=x{scale} | AMP={use_amp}\n"
        f"End-to-end: {total_seconds:.3f}s ({image_count / max(total_seconds, 1e-9):.2f} images/s)\n"
        f"Model execution: {model_seconds:.3f}s ({image_count / max(model_seconds, 1e-9):.2f} images/s)"
    )


if __name__ == "__main__":
    main()
