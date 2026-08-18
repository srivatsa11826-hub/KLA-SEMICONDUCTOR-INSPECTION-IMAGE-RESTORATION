#!/usr/bin/env python3
"""Standalone mixed-precision training entry point."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import SyntheticDegradation
from src.dataset import RestorationDataset
from src.losses import CompositeRestorationLoss
from src.metrics import MetricAccumulator, RestorationMetrics
from src.models.init import build_model
from src.utils import (
    AverageMeter,
    CSVLogger,
    atomic_torch_save,
    checkpoint_payload,
    count_parameters,
    extract_state_dict,
    load_yaml,
    resolve_device,
    save_yaml,
    seed_worker,
    set_seed,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KLA image restoration model")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Resume full training state from checkpoint")
    parser.add_argument("--pretrained", default=None, help="Load model weights only and start a new run")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = config.setdefault("training", {})
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.lr is not None:
        training["learning_rate"] = args.lr
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.compile is not None:
        training["compile"] = args.compile


def make_dataset(
    config: Mapping[str, Any], split: str, training: bool
) -> RestorationDataset:
    data_config = config["data"]
    split_config = data_config[split]
    synthetic = SyntheticDegradation.from_config(data_config.get("synthetic", {}))
    return RestorationDataset(
        gt_dir=split_config["gt_dir"],
        noisy_dir=split_config.get("noisy_dir"),
        channels=int(data_config.get("channels", config["model"].get("img_channel", 3))),
        gt_patch_size=split_config.get("gt_patch_size"),
        training=training,
        augment=bool(split_config.get("augment", training)),
        synthetic_degradation=synthetic,
        expected_scale=int(config["model"].get("scale", 1)),
        seed=int(config.get("seed", 42)) + (0 if training else 1_000_003),
        deterministic_synthetic=bool(split_config.get("deterministic_synthetic", not training)),
    )


def make_loader(
    dataset: RestorationDataset,
    training: bool,
    config: Mapping[str, Any],
) -> DataLoader:
    training_config = config["training"]
    workers = int(training_config.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(training_config.get("batch_size", 8)),
        "shuffle": training,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": bool(training_config.get("drop_last", True)) if training else False,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(int(config.get("seed", 42)) + (0 if training else 1)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(training_config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(training_config.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(
    batch: Mapping[str, Any], device: torch.device, channels_last: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    if channels_last and inputs.ndim == 4:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
        targets = targets.contiguous(memory_format=torch.channels_last)
    return inputs, targets


def train_one_epoch(
    model: nn.Module,
    criterion: CompositeRestorationLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
    config: Mapping[str, Any],
) -> dict[str, float]:
    model.train()
    criterion.train()
    training_config = config["training"]
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    channels_last = bool(training_config.get("channels_last", True)) and device.type == "cuda"
    accumulation = max(1, int(training_config.get("accumulation_steps", 1)))
    grad_clip = float(training_config.get("grad_clip_norm", 0.0))
    meters = {name: AverageMeter() for name in ("total", "charbonnier", "ms_ssim_loss", "lpips_loss")}

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch:03d}", dynamic_ncols=True)
    final_group_size = len(loader) % accumulation or accumulation
    for step, batch in enumerate(progress):
        inputs, targets = move_batch(batch, device, channels_last)
        # Do not under-scale the final partial accumulation group.
        divisor = (
            final_group_size
            if step >= len(loader) - final_group_size
            else accumulation
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            prediction = model(inputs)
            loss, parts = criterion(prediction, targets)
            scaled_loss = loss / divisor
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = inputs.shape[0]
        for name, value in parts.items():
            meters[name].update(float(value.detach().item()), batch_size)
        progress.set_postfix(loss=f"{meters['total'].average:.4f}")
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: CompositeRestorationLoss,
    metrics: RestorationMetrics,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    config: Mapping[str, Any],
) -> dict[str, float]:
    model.eval()
    criterion.eval()
    metrics.eval()
    training_config = config["training"]
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    channels_last = bool(training_config.get("channels_last", True)) and device.type == "cuda"
    loss_meter = AverageMeter()
    accumulator = MetricAccumulator()
    progress = tqdm(loader, desc=f"valid {epoch:03d}", dynamic_ncols=True)
    for batch in progress:
        inputs, targets = move_batch(batch, device, channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            prediction = model(inputs)
            loss, _ = criterion(prediction, targets)
        batch_size = inputs.shape[0]
        loss_meter.update(float(loss.item()), batch_size)
        values = metrics(prediction, targets)
        accumulator.update(values, batch_size)
        progress.set_postfix(psnr=f"{values['psnr']:.2f}", ssim=f"{values['ssim']:.4f}")
    result = accumulator.compute()
    result["loss"] = loss_meter.average
    return result


def validation_score(metrics: Mapping[str, float], config: Mapping[str, Any]) -> float:
    weights = config.get("validation", {}).get("score_weights", {})
    return (
        float(weights.get("psnr", 0.1)) * metrics.get("psnr", 0.0)
        + float(weights.get("ssim", 1.0)) * metrics.get("ssim", 0.0)
        - float(weights.get("lpips", 1.0)) * metrics.get("lpips", 0.0)
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    apply_overrides(config, args)
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=args.deterministic)
    device = resolve_device(args.device)

    experiment = config.get("experiment", {})
    run_dir = Path(experiment.get("output_dir", "results")) / experiment.get("name", "experiment")
    checkpoint_dir = run_dir / "checkpoints"
    weights_dir = Path(experiment.get("weights_dir", "weights"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "resolved_config.yaml")

    train_dataset = make_dataset(config, "train", training=True)
    val_dataset = make_dataset(config, "val", training=False)
    train_loader = make_loader(train_dataset, True, config)
    val_loader = make_loader(val_dataset, False, config)
    if len(train_loader) == 0:
        raise RuntimeError("Training loader has zero batches; reduce batch_size or disable drop_last")

    model = build_model(config["model"]).to(device)
    channels_last = bool(config["training"].get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = CompositeRestorationLoss.from_config(config.get("loss", {})).to(device)
    compute_val_lpips = bool(config.get("validation", {}).get("compute_lpips", True))
    metric_suite = RestorationMetrics(
        compute_lpips=compute_val_lpips,
        lpips_net=config.get("loss", {}).get("lpips_net", "alex"),
        lpips_random_backbone=bool(config.get("loss", {}).get("lpips_random_backbone", False)),
        # Share the already-loaded LPIPS network with the loss to save memory.
        lpips_model=criterion.lpips if compute_val_lpips else None,
    ).to(device)

    training_config = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    epochs = int(training_config.get("epochs", 200))
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training_config.get("min_learning_rate", 1e-6)),
    )
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(amp)
    start_epoch = 1
    best = {"score": -math.inf, "psnr": -math.inf, "ssim": -math.inf}

    if args.resume:
        checkpoint = torch_load(args.resume, map_location="cpu")
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        if isinstance(checkpoint, Mapping):
            if checkpoint.get("optimizer"):
                optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scheduler"):
                scheduler.load_state_dict(checkpoint["scheduler"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best.update(checkpoint.get("best", {}))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    if bool(training_config.get("compile", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")
        model = torch.compile(model, mode="default")

    print(
        f"Device={device} | train={len(train_dataset)} | val={len(val_dataset)} | "
        f"parameters={count_parameters(model):,} | AMP={amp}"
    )
    logger = CSVLogger(
        run_dir / "history.csv",
        ["epoch", "lr", "train_loss", "val_loss", "psnr", "ssim", "lpips", "score", "seconds"],
    )
    try:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard = SummaryWriter(run_dir / "tensorboard")
    except ImportError:
        tensorboard = None

    validation_every = max(1, int(config.get("validation", {}).get("every", 1)))
    save_every = max(1, int(training_config.get("save_every", 10)))
    latest_val: dict[str, float] = {}
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch, config
        )
        scheduler.step()

        latest_val = {}
        if epoch % validation_every == 0 or epoch == epochs:
            latest_val = validate(model, criterion, metric_suite, val_loader, device, epoch, config)
            score = validation_score(latest_val, config)
        else:
            score = float("nan")

        improved_score = improved_psnr = improved_ssim = False
        if latest_val:
            improved_score = score > best["score"]
            improved_psnr = latest_val["psnr"] > best["psnr"]
            improved_ssim = latest_val["ssim"] > best["ssim"]
            if improved_score:
                best["score"] = score
            if improved_psnr:
                best["psnr"] = latest_val["psnr"]
            if improved_ssim:
                best["ssim"] = latest_val["ssim"]

        # Build checkpoints after all monitor values are updated, so every file
        # contains resume-safe best-metric metadata.
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, epoch, config, best)
        atomic_torch_save(payload, checkpoint_dir / "last.pth")
        if epoch % save_every == 0:
            atomic_torch_save(payload, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        if improved_score:
            atomic_torch_save(payload, checkpoint_dir / "best_model.pth")
            atomic_torch_save(payload, weights_dir / "best_model.pth")
        if improved_psnr:
            atomic_torch_save(payload, checkpoint_dir / "best_psnr.pth")
        if improved_ssim:
            atomic_torch_save(payload, checkpoint_dir / "best_ssim.pth")

        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["total"],
            "val_loss": latest_val.get("loss", ""),
            "psnr": latest_val.get("psnr", ""),
            "ssim": latest_val.get("ssim", ""),
            "lpips": latest_val.get("lpips", ""),
            "score": score if not math.isnan(score) else "",
            "seconds": elapsed,
        }
        logger.log(row)
        if tensorboard is not None:
            tensorboard.add_scalar("loss/train", train_stats["total"], epoch)
            for key, value in latest_val.items():
                tensorboard.add_scalar(f"validation/{key}", value, epoch)
            tensorboard.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch)
        psnr_text = f"{latest_val['psnr']:.3f}" if "psnr" in latest_val else "N/A"
        ssim_text = f"{latest_val['ssim']:.5f}" if "ssim" in latest_val else "N/A"
        lpips_text = f"{latest_val['lpips']:.5f}" if "lpips" in latest_val else "N/A"
        print(
            f"Epoch {epoch:03d}/{epochs}: train={train_stats['total']:.5f} "
            f"PSNR={psnr_text} SSIM={ssim_text} LPIPS={lpips_text} ({elapsed:.1f}s)"
        )

    if tensorboard is not None:
        tensorboard.close()
    print(f"Training complete. Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
