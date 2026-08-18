#!/usr/bin/env python3
"""Generate paired procedural semiconductor GT/NoisyLR restoration data."""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

FIELDS = [
    "filename", "split", "seed", "gt_height", "gt_width", "lr_height", "lr_width",
    "scale", "pattern_type", "defect_types", "defect_count", "difficulty",
    "gaussian_sigma", "speckle_sigma", "interpolation", "degradation_order",
    "gt_min", "gt_max", "noisylr_min", "noisylr_max",
]


def smooth_background(size: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(-1, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    angle = rng.uniform(0, 2 * math.pi)
    gradient = rng.uniform(0.02, 0.09) * (
        xx * math.cos(angle) + yy * math.sin(angle)
    )
    low = rng.uniform(-0.035, 0.035, (8, 8)).astype(np.float32)
    low = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    radial = rng.uniform(-0.025, 0.025) * (xx**2 + yy**2)
    return np.clip(rng.uniform(0.10, 0.34) + gradient + low + radial, 0.04, 0.46).astype(np.float32)


def draw_line_space(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape
    orientation = rng.choice(("horizontal", "vertical", "both"))
    for vertical in ((True, False) if orientation == "both" else (orientation == "vertical",)):
        axis_size = w if vertical else h
        pitch = int(rng.integers(6, max(7, axis_size // 9)))
        width = max(1, int(pitch * rng.uniform(0.25, 0.62)))
        offset = int(rng.integers(0, pitch))
        intensity = float(rng.uniform(0.58, 0.97))
        for position in range(offset, axis_size, pitch):
            if vertical:
                image[:, position : min(w, position + width)] = intensity
            else:
                image[position : min(h, position + width), :] = intensity


def draw_routing(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape
    for _ in range(int(rng.integers(2, 5))):
        vertical = bool(rng.integers(0, 2))
        limit = w if vertical else h
        width = int(rng.integers(2, max(4, min(h, w) // 35)))
        count = min(int(rng.integers(4, 13)), limit - width)
        positions = rng.choice(limit - width, count, replace=False)
        intensity = float(rng.uniform(0.52, 0.96))
        for position in positions:
            if vertical:
                image[:, position : position + width] = intensity
            else:
                image[position : position + width, :] = intensity
        # Add a few orthogonal jogs rather than only infinite straight lines.
        for _ in range(int(rng.integers(1, 4))):
            x1, x2 = sorted(rng.integers(0, w, 2))
            y1, y2 = sorted(rng.integers(0, h, 2))
            if vertical:
                cv2.line(image, (x1, y1), (x2, y1), intensity, width)
            else:
                cv2.line(image, (x1, y1), (x1, y2), intensity, width)


def draw_vias(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape
    rows, cols = int(rng.integers(4, 13)), int(rng.integers(4, 13))
    radius = int(rng.integers(2, max(3, min(h, w) // 25)))
    circle = bool(rng.integers(0, 2))
    intensity = float(rng.uniform(0.03, 0.16) if rng.random() < 0.5 else rng.uniform(0.80, 1.0))
    row_step, col_step = h // (rows + 1), w // (cols + 1)
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cy = int(np.clip(row * row_step + rng.integers(-1, 2), radius, h - radius - 1))
            cx = int(np.clip(col * col_step + rng.integers(-1, 2), radius, w - radius - 1))
            if circle:
                cv2.circle(image, (cx, cy), radius, intensity, -1)
            else:
                image[cy - radius : cy + radius, cx - radius : cx + radius] = intensity


def draw_cells(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape
    for _ in range(int(rng.integers(3, 9))):
        pw, ph = int(rng.integers(w // 9, w // 4)), int(rng.integers(h // 9, h // 4))
        x, y = int(rng.integers(0, w - pw + 1)), int(rng.integers(0, h - ph + 1))
        image[y : y + ph, x : x + pw] = rng.uniform(0.56, 0.91)
        inset = int(rng.integers(2, max(3, min(pw, ph) // 5)))
        cx, cy = x + pw // 2, y + ph // 2
        image[cy - inset : cy + inset, cx - inset : cx + inset] = rng.uniform(0.08, 0.30)


def draw_diagonals(image: np.ndarray, rng: np.random.Generator) -> None:
    h, w = image.shape
    for _ in range(int(rng.integers(3, 9))):
        points = rng.integers((0, 0), (w, h), size=(3, 2), endpoint=False).astype(np.int32)
        cv2.polylines(
            image, [points.reshape(-1, 1, 2)], False,
            float(rng.uniform(0.62, 0.93)), int(rng.integers(2, 6)), cv2.LINE_AA,
        )


def inject_defects(image: np.ndarray, rng: np.random.Generator) -> list[str]:
    h, w = image.shape
    types = ("particle", "scratch", "broken_line", "bridge", "missing_via", "edge_roughness")
    selected = rng.choice(types, int(rng.choice((1, 2, 3), p=(0.50, 0.35, 0.15))), replace=True).tolist()
    for kind in selected:
        if kind == "particle":
            cx, cy = int(rng.integers(8, w - 8)), int(rng.integers(8, h - 8))
            cv2.circle(image, (cx, cy), int(rng.integers(2, 6)), float(rng.choice((0.03, 0.97))), -1)
        elif kind == "scratch":
            x, y = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
            length, angle = int(rng.integers(15, 48)), rng.uniform(0, 2 * math.pi)
            end = (int(x + length * math.cos(angle)), int(y + length * math.sin(angle)))
            cv2.line(image, (x, y), end, float(rng.choice((0.05, 0.95))), int(rng.integers(1, 3)))
        elif kind in ("broken_line", "missing_via"):
            x, y = int(rng.integers(8, w - 18)), int(rng.integers(8, h - 18))
            pw, ph = int(rng.integers(4, 11)), int(rng.integers(4, 11))
            image[y : y + ph, x : x + pw] = rng.uniform(0.12, 0.34)
        elif kind == "bridge":
            x, y = int(rng.integers(8, w - 22)), int(rng.integers(8, h - 14))
            image[y : y + int(rng.integers(3, 7)), x : x + int(rng.integers(7, 17))] = rng.uniform(0.76, 0.96)
        else:
            x, y = int(rng.integers(5, w - 20)), int(rng.integers(5, h - 20))
            patch = image[y : y + 15, x : x + 15]
            image[y : y + 15, x : x + 15] = np.clip(
                patch + rng.uniform(-0.14, 0.14, patch.shape), 0, 1
            )
    return selected


def clean_scene(size: int, rng: np.random.Generator) -> tuple[np.ndarray, str, list[str]]:
    image = smooth_background(size, rng)
    pattern = str(rng.choice(("dense_interconnects", "crossed_routing", "via_array", "transistor_cell", "diagonal_hybrid")))
    if pattern == "dense_interconnects":
        draw_line_space(image, rng)
    elif pattern == "crossed_routing":
        draw_routing(image, rng)
    elif pattern == "via_array":
        draw_routing(image, rng); draw_vias(image, rng)
    elif pattern == "transistor_cell":
        draw_cells(image, rng); draw_vias(image, rng)
    else:
        draw_diagonals(image, rng); draw_line_space(image, rng)
    kernel = int(rng.choice((3, 5)))
    image = cv2.GaussianBlur(image, (kernel, kernel), rng.uniform(0.4, 0.9))
    defects = inject_defects(image, rng)
    image = np.rot90(image, int(rng.integers(0, 4)))
    if rng.random() < 0.5: image = np.fliplr(image)
    if rng.random() < 0.5: image = np.flipud(image)
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32), pattern, defects


def degrade(image: np.ndarray, scale: int, difficulty: str, rng: np.random.Generator):
    ranges = {
        "easy": ((0.005, 0.025), (0.010, 0.040)),
        "medium": ((0.025, 0.055), (0.040, 0.090)),
        "difficult": ((0.055, 0.080), (0.090, 0.150)),
    }
    g_range, s_range = ranges[difficulty]
    sigma_g, sigma_s = rng.uniform(*g_range), rng.uniform(*s_range)
    interpolation = str(rng.choice(("bilinear", "bicubic")))
    operations = ["gaussian", "speckle", "downsample"]
    rng.shuffle(operations)
    output = image.copy()
    for operation in operations:
        if operation == "gaussian":
            output += rng.normal(0, sigma_g, output.shape).astype(np.float32)
        elif operation == "speckle":
            output += output * rng.normal(0, sigma_s, output.shape).astype(np.float32)
        else:
            mode = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_CUBIC
            output = cv2.resize(output, (output.shape[1] // scale, output.shape[0] // scale), interpolation=mode)
    return output.astype(np.float32), {
        "gaussian_sigma": float(sigma_g), "speckle_sigma": float(sigma_s),
        "interpolation": interpolation, "degradation_order": ">".join(operations),
    }


def preview(samples: list[dict], path: Path) -> None:
    rows = []
    for sample in samples[:12]:
        gt = np.rint(sample["gt"] * 255).astype(np.uint8)
        lr = np.rint(np.clip(sample["lr"], 0, 1) * 255).astype(np.uint8)
        nearest = cv2.resize(lr, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        bicubic = cv2.resize(lr, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
        panels = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (gt, nearest, bicubic)]
        for panel, label in zip(panels, ("Clean GT", "NoisyLR nearest", "NoisyLR bicubic")):
            cv2.putText(panel, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 255, 255), 1)
        info = np.full((gt.shape[0], 250, 3), 28, np.uint8)
        lines = (sample["pattern"], sample["order"], f"g={sample['g']:.4f} s={sample['s']:.4f}", f"range={sample['lr'].min():.3f},{sample['lr'].max():.3f}")
        for index, text in enumerate(lines):
            cv2.putText(info, str(text), (7, 24 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1)
        rows.append(np.hstack([*panels, info]))
    if rows and not cv2.imwrite(str(path), np.vstack(rows)):
        raise OSError(f"Could not write preview: {path}")


def generate_split(name: str, count: int, seed_offset: int, args, previews: list[dict]) -> None:
    gt_dir, lr_dir = args.output_dir / name / "GT", args.output_dir / name / "NoisyLR"
    gt_dir.mkdir(parents=True, exist_ok=True); lr_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    difficulty = np.array(["easy"] * round(count * .20) + ["medium"] * round(count * .55))
    difficulty = np.concatenate([difficulty, np.array(["difficult"] * (count - len(difficulty)))])
    np.random.default_rng(seed_offset - 1).shuffle(difficulty)
    for index in tqdm(range(count), desc=f"generate {name}"):
        item_seed = seed_offset + index
        rng = np.random.default_rng(item_seed)
        gt, pattern, defects = clean_scene(args.gt_size, rng)
        lr, degradation = degrade(gt, args.scale, str(difficulty[index]), rng)
        # Prefixes make IDs globally unique while names still match within each pair.
        sample_id = f"{name}_{index + 1:06d}"
        if not cv2.imwrite(str(gt_dir / f"{sample_id}.png"), np.rint(gt * 65535).astype(np.uint16)):
            raise OSError(f"Failed to write GT {sample_id}")
        np.save(lr_dir / f"{sample_id}.npy", lr, allow_pickle=False)
        metadata.append({
            "filename": sample_id, "split": name, "seed": item_seed,
            "gt_height": gt.shape[0], "gt_width": gt.shape[1],
            "lr_height": lr.shape[0], "lr_width": lr.shape[1], "scale": args.scale,
            "pattern_type": pattern, "defect_types": ";".join(defects),
            "defect_count": len(defects), "difficulty": str(difficulty[index]),
            "gaussian_sigma": f"{degradation['gaussian_sigma']:.7f}",
            "speckle_sigma": f"{degradation['speckle_sigma']:.7f}",
            "interpolation": degradation["interpolation"],
            "degradation_order": degradation["degradation_order"],
            "gt_min": f"{gt.min():.7f}", "gt_max": f"{gt.max():.7f}",
            "noisylr_min": f"{lr.min():.7f}", "noisylr_max": f"{lr.max():.7f}",
        })
        if len(previews) < 12 and index % max(1, count // 5) == 0:
            previews.append({"gt": gt, "lr": lr, "pattern": pattern,
                             "order": degradation["degradation_order"],
                             "g": degradation["gaussian_sigma"], "s": degradation["speckle_sigma"]})
    with (args.output_dir / "metadata" / f"{name}_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(metadata)


def dataset_readme(args) -> str:
    return f"""# Procedural Semiconductor Restoration Dataset

Generated deterministically with seed {args.seed}.

- GT: 16-bit grayscale PNG, normalized domain [0, 1], {args.gt_size}x{args.gt_size}
- NoisyLR: unclipped float32 NPY, {args.gt_size // args.scale}x{args.gt_size // args.scale}
- Scale: x{args.scale}
- Counts: train={args.train_count}, val={args.val_count}, test={args.test_count}

Regenerate:
```bash
python generate_dataset.py --output_dir semiconductor_dataset --train_count {args.train_count} --val_count {args.val_count} --test_count {args.test_count} --gt_size {args.gt_size} --scale {args.scale} --seed {args.seed} --overwrite
```

Validate:
```bash
python validate_dataset.py --dataset_dir semiconductor_dataset --scale {args.scale}
```

Load:
```python
import cv2, numpy as np
gt = cv2.imread('train/GT/train_000001.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
lr = np.load('train/NoisyLR/train_000001.npy', allow_pickle=False)
```
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("semiconductor_dataset"))
    parser.add_argument("--train_count", type=int, default=1000)
    parser.add_argument("--val_count", type=int, default=150)
    parser.add_argument("--test_count", type=int, default=150)
    parser.add_argument("--gt_size", type=int, default=256, choices=(256, 512))
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_count, args.val_count, args.test_count) < 1:
        raise ValueError("All split counts must be positive")
    if args.scale < 1 or args.gt_size % args.scale:
        raise ValueError("gt_size must be exactly divisible by a positive scale")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    (args.output_dir / "metadata").mkdir(parents=True)
    started = time.perf_counter(); previews: list[dict] = []
    for split, count, offset in (
        ("train", args.train_count, args.seed),
        ("val", args.val_count, args.seed + 100_000),
        ("test", args.test_count, args.seed + 200_000),
    ):
        generate_split(split, count, offset, args, previews)
    preview(previews, args.output_dir / "preview.png")
    (args.output_dir / "README.md").write_text(dataset_readme(args), encoding="utf-8")
    for source in (Path(__file__), Path(__file__).with_name("validate_dataset.py")):
        shutil.copy2(source, args.output_dir / source.name)

    # Validate before packaging; a bad dataset is never presented as complete.
    sys.path.insert(0, str(Path(__file__).parent))
    from validate_dataset import validate_dataset
    report = validate_dataset(args.output_dir, args.scale)
    if not args.no_zip:
        zip_path = args.output_dir.with_suffix(".zip")
        if zip_path.exists(): zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in args.output_dir.rglob("*"):
                if path.is_file(): archive.write(path, Path(args.output_dir.name) / path.relative_to(args.output_dir))
        print(f"ZIP: {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MiB)")
    elapsed = time.perf_counter() - started
    total = args.train_count + args.val_count + args.test_count
    print(f"Generated and validated {total} pairs in {elapsed:.2f}s ({total / elapsed:.2f} pairs/s)")
    print(report)


if __name__ == "__main__":
    main()
