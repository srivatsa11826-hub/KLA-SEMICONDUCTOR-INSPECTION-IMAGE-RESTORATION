#!/usr/bin/env python3
"""Strict integrity validator for procedural semiconductor restoration data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
EXPECTED_OPERATIONS = {"gaussian", "speckle", "downsample"}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def file_map(directory: Path, suffix: str) -> dict[str, Path]:
    files = list(directory.glob(f"*{suffix}"))
    mapping = {path.stem: path for path in files}
    require(len(mapping) == len(files), f"Duplicate stems in {directory}")
    return mapping


def validate_dataset(dataset_dir: str | Path, scale: int = 2) -> dict[str, Any]:
    root = Path(dataset_dir)
    require(root.is_dir(), f"Dataset directory does not exist: {root}")
    require(scale >= 1, "scale must be positive")
    counts: dict[str, int] = {}
    out_of_range = 0
    total = 0
    global_ids: set[str] = set()
    content_hashes: set[str] = set()

    for split in SPLITS:
        gt_dir, lr_dir = root / split / "GT", root / split / "NoisyLR"
        metadata_path = root / "metadata" / f"{split}_metadata.csv"
        require(gt_dir.is_dir(), f"Missing directory: {gt_dir}")
        require(lr_dir.is_dir(), f"Missing directory: {lr_dir}")
        require(metadata_path.is_file(), f"Missing metadata: {metadata_path}")
        gt_files, lr_files = file_map(gt_dir, ".png"), file_map(lr_dir, ".npy")
        require(gt_files.keys() == lr_files.keys(), f"Pair-key mismatch in {split}")
        require(bool(gt_files), f"Split is empty: {split}")

        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        metadata = {row.get("filename", ""): row for row in rows}
        require(len(metadata) == len(rows), f"Duplicate/empty metadata IDs in {split}")
        require(metadata.keys() == gt_files.keys(), f"Metadata/file ID mismatch in {split}")
        counts[split] = len(gt_files); total += len(gt_files)

        batch_examples = []
        for sample_id in sorted(gt_files):
            require(sample_id not in global_ids, f"ID reused across splits: {sample_id}")
            global_ids.add(sample_id)
            gt_path, lr_path, row = gt_files[sample_id], lr_files[sample_id], metadata[sample_id]
            require(row["split"] == split, f"Wrong metadata split for {sample_id}")
            require(int(row["scale"]) == scale, f"Wrong metadata scale for {sample_id}")
            operations = row["degradation_order"].split(">")
            require(len(operations) == 3 and set(operations) == EXPECTED_OPERATIONS,
                    f"Invalid degradation order for {sample_id}: {operations}")

            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            require(gt is not None, f"Unreadable GT: {gt_path}")
            require(gt.dtype == np.uint16 and gt.ndim == 2, f"GT must be 2D uint16: {gt_path}")
            try:
                lr = np.load(lr_path, allow_pickle=False)
            except Exception as error:
                raise ValidationError(f"Unreadable NPY: {lr_path}: {error}") from error
            require(lr.dtype == np.float32 and lr.ndim == 2, f"NoisyLR must be 2D float32: {lr_path}")
            require(gt.shape == (lr.shape[0] * scale, lr.shape[1] * scale),
                    f"Scale mismatch for {sample_id}: GT={gt.shape}, LR={lr.shape}")
            require(np.isfinite(lr).all(), f"NaN/Inf in {lr_path}")
            require(float(np.std(gt.astype(np.float32))) > 1e-3, f"Constant GT: {gt_path}")
            require(float(np.std(lr)) > 1e-5, f"Constant NoisyLR: {lr_path}")
            if float(lr.min()) < 0 or float(lr.max()) > 1: out_of_range += 1
            require(int(row["gt_height"]) == gt.shape[0] and int(row["gt_width"]) == gt.shape[1],
                    f"Wrong GT metadata shape for {sample_id}")
            require(int(row["lr_height"]) == lr.shape[0] and int(row["lr_width"]) == lr.shape[1],
                    f"Wrong LR metadata shape for {sample_id}")
            require(abs(float(row["noisylr_min"]) - float(lr.min())) < 2e-6,
                    f"Wrong LR minimum metadata for {sample_id}")
            require(abs(float(row["noisylr_max"]) - float(lr.max())) < 2e-6,
                    f"Wrong LR maximum metadata for {sample_id}")
            digest = hashlib.sha256(gt.tobytes()).hexdigest()
            require(digest not in content_hashes, f"Exact GT content duplicated across splits/samples: {sample_id}")
            content_hashes.add(digest)
            if len(batch_examples) < 16: batch_examples.append(lr)

        batch = np.stack(batch_examples)
        require(batch.ndim == 3 and batch.shape[0] == min(16, len(gt_files)),
                f"Batch stacking failed in {split}")

    require(out_of_range > 0, "No NoisyLR sample contains out-of-range pixels")
    report = {
        "status": "passed", "counts": counts, "total_pairs": total,
        "out_of_range_samples": out_of_range,
        "out_of_range_percent": round(100 * out_of_range / total, 3),
        "invalid_pairs": 0,
    }
    print("\nDataset validation passed")
    print(f"Train pairs: {counts['train']}")
    print(f"Validation pairs: {counts['val']}")
    print(f"Test pairs: {counts['test']}")
    print(f"NoisyLR samples with out-of-range pixels: {report['out_of_range_percent']:.3f}%")
    print("Invalid pairs: 0")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("semiconductor_dataset"))
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = validate_dataset(args.dataset_dir, args.scale)
    except ValidationError as error:
        print(f"VALIDATION FAILURE: {error}")
        raise SystemExit(1)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
