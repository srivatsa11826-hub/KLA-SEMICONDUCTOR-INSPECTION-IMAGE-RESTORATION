# KLA Semiconductor Image Restoration Pipeline

A modular PyTorch 2.x pipeline for joint denoising and super-resolution of semiconductor inspection imagery. It preserves out-of-range floating-point `NoisyLR` values at model input, trains a low-resolution NAFNet-derived backbone, reconstructs with `PixelShuffle`, and clamps only final evaluation/inference outputs to `[0, 1]`.

For the shortest setup and inference instructions, start with [`QUICKSTART.md`](QUICKSTART.md).

## Highlights

- Strict `.png`/`.npy` NoisyLR-to-GT pairing by relative path
- Real paired data **or** on-the-fly Gaussian + speckle + downsampling synthesis
- Gaussian, multiplicative speckle, and bilinear/bicubic downsampling in random order
- NAFNet blocks plus explicit low/high-frequency LR feature fusion
- Lightweight U-Net comparison baseline
- Charbonnier + adaptive MS-SSIM + LPIPS composite objective
- PSNR, SSIM, and LPIPS validation; best composite/PSNR/SSIM checkpoints
- CUDA AMP, channels-last tensors, optional `torch.compile`, pinned-memory loading
- Variable-size batching, overlap-tiled inference, and asynchronous image writing
- CSV and TensorBoard experiment logs

## Repository layout

```text
semicon_restoration/
├── configs/
│   ├── train_config.yaml
│   ├── synthetic_cpu_train.yaml
│   └── real_kla_finetune.yaml
├── tools/
│   ├── generate_dataset.py
│   └── validate_dataset.py
├── src/
│   ├── init.py
│   ├── dataset.py
│   ├── augmentations.py
│   ├── models/
│   │   ├── init.py
│   │   ├── nafnet.py
│   │   └── baseline_unet.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
├── tests/test_core.py
├── weights/
│   ├── best_model.pth
│   └── real_kla/best_model.pth
├── results/
│   ├── real_kla_finetune_x2/
│   └── real_kla_restored_test.zip
├── train.py
├── inference.py
├── requirements.txt
├── TEST_REPORT.md
└── README.md
```

## Preferred checkpoint: fine-tuned on the supplied Drive data

The public Drive archives were downloaded and audited, then 3,200 pairs were split deterministically into 2,880 training and 320 validation pairs. After four fine-tuning epochs, the preferred checkpoint is:

```text
weights/real_kla/best_model.pth
```

Held-out results:

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 22.9724 | 0.52377 | 0.44160 |
| Fine-tuned NAFNet | **26.5442** | **0.67932** | **0.37290** |

All 400 supplied test inputs were restored into `results/real_kla_restored_test.zip`. See `results/real_kla_finetune_x2/DRIVE_TRAINING_REPORT.md` for hashes, data audit, training history, and evaluation details.

A synthetic-only checkpoint remains available at `weights/best_model.pth` for comparison.

## 1. Environment setup

Python 3.10–3.13 is supported by the pinned dependency set.

```bash
cd semicon_restoration
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a specific CUDA runtime, install the matching official PyTorch wheel first, then install the remaining requirements. Example for CUDA 12.8:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only verification, replace `cu128` with `cpu`. Choose the CUDA index supported by the NVIDIA driver on the target machine; the rest of the code and dependency versions remain unchanged.

`lpips` may download an AlexNet backbone on first use. On an offline machine, pre-populate the PyTorch cache. `loss.lpips_random_backbone: true` is provided only for offline smoke tests; a random backbone is not recommended for a real challenge run.

## 2. Data layout

Paired training is expected to use matching relative paths (the extensions may differ):

```text
data/
├── train/
│   ├── NoisyLR/wafer_a/0001.npy
│   └── GT/wafer_a/0001.png
└── val/
    ├── NoisyLR/0001.npy
    └── GT/0001.png
```

Every GT dimension must be the same integer multiple of the corresponding NoisyLR dimension. The multiple must equal `model.scale`. A floating-point NPY input is cast to `float32` but is **not** clipped or rescaled. Integer PNG/NPY data is divided by the dtype maximum. GT tensors are clipped to `[0, 1]`.

To train from GT-only images, set a split's `noisy_dir: null`. `SyntheticDegradation` then applies all three challenge degradations once in a random permutation. Keep `data.synthetic.scale` fixed and equal to `model.scale` when batching.

## 3. Configuration

Edit `configs/train_config.yaml`:

- `data.train` / `data.val`: dataset roots and HR crop size
- `data.synthetic`: noise ranges, interpolation modes, and scale
- `model`: `nafnet` or `baseline_unet`, widths/depths, and scale
- `loss`: default weights are `1.0` Charbonnier, `0.5` MS-SSIM, `0.1` LPIPS
- `training`: optimizer, cosine schedule, workers, AMP, and compile settings
- `validation.score_weights`: score = `w_psnr*PSNR + w_ssim*SSIM - w_lpips*LPIPS`

A 256 HR crop at scale x2 produces a 128 LR input. NAFNet performs its expensive feature extraction at that LR resolution.

## 4. Training

```bash
python train.py --config configs/train_config.yaml --device cuda
```

CLI overrides:

```bash
python train.py \
  --config configs/train_config.yaml \
  --epochs 300 --lr 1e-4 --batch_size 8 --seed 123 \
  --device cuda --compile
```

Resume the complete optimizer/scheduler state:

```bash
python train.py --config configs/train_config.yaml \
  --resume results/nafnet_kla_x2/checkpoints/last.pth --device cuda
```

Fine-tune from model weights while starting a fresh optimizer and epoch count:

```bash
python train.py --config configs/train_config.yaml \
  --pretrained weights/best_model.pth --device cuda
```

Artifacts are written under `results/<experiment.name>/`:

- `resolved_config.yaml`
- `history.csv`
- `tensorboard/`
- `checkpoints/last.pth`
- `checkpoints/best_model.pth` (weighted challenge-style score)
- `checkpoints/best_psnr.pth`
- `checkpoints/best_ssim.pth`
- `weights/best_model.pth` (mirrored best composite checkpoint for the default inference command)

Launch TensorBoard with:

```bash
tensorboard --logdir results/nafnet_kla_x2/tensorboard
```

## 5. Inference

The required command is supported directly:

```bash
python inference.py \
  --input_dir <path_to_input_folder> \
  --output_dir <path_to_output_folder> \
  --weights weights/best_model.pth \
  --batch_size 8 \
  --device cuda
```

For images from the supplied Drive distribution, replace the weights argument with:

```bash
--weights weights/real_kla/best_model.pth
```

The checkpoint's embedded architecture config takes precedence over `--config`. Raw state dictionaries use the architecture from `--config`.

Useful options:

```bash
# 16-bit PNG output, compiled CUDA graph path
python inference.py ... --bit_depth 16 --compile --amp --channels_last

# Memory-limited inference; tile size and overlap are LR pixels
python inference.py ... --tile_size 256 --tile_overlap 32

# Preserve NPY inputs as float32 NPY outputs, PNG inputs as PNG
python inference.py ... --save_format auto
```

Input subdirectories are reproduced in the output directory. Mixed image dimensions are grouped by tensor shape inside each loader batch. Final predictions are always clamped to `[0, 1]` before either PNG quantization or NPY saving. The script reports both model-only CUDA-event timing and true end-to-end throughput from the first disk read through the completion of asynchronous writes.

## 6. Smoke test without a dataset

Create tiny synthetic paired data:

```bash
python - <<'PY'
from pathlib import Path
import cv2, numpy as np
for split in ('train', 'val'):
    for folder in ('GT', 'NoisyLR'):
        Path(f'data/{split}/{folder}').mkdir(parents=True, exist_ok=True)
    for i in range(4):
        gt = np.random.default_rng(i).random((256, 256, 3), dtype=np.float32)
        lr = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_CUBIC)
        lr += np.random.default_rng(i + 10).normal(0, .03, lr.shape).astype('float32')
        cv2.imwrite(f'data/{split}/GT/{i:03d}.png', cv2.cvtColor((gt*255).astype('uint8'), cv2.COLOR_RGB2BGR))
        np.save(f'data/{split}/NoisyLR/{i:03d}.npy', lr)
PY
```

For a quick CPU check, set `epochs: 1`, `batch_size: 1`, `num_workers: 0`, `gt_patch_size: 64`, `lambda_lpips: 0`, and `validation.compute_lpips: false`, then run:

```bash
python train.py --device cpu --epochs 1 --batch_size 1
```

## Design notes

The implementation is an original compact integration of concepts from [NAFNet](https://github.com/megvii-research/NAFNet), [SwinIR](https://github.com/JingyunLiang/SwinIR), [Restormer](https://github.com/swz30/Restormer), and [BasicSR](https://github.com/XPixelGroup/BasicSR). It does not require cloning those repositories. The NAF backbone is activation-free; the only nonlinearities in the primary model arise from multiplication gates, normalization, attention multiplication, and reconstruction arithmetic.

For leaderboard work, tune crop size, model width/depth, synthetic noise ranges, score weights, and inference precision against the actual validation distribution. Keep final metric and submission outputs clamped, but do not clamp floating NoisyLR tensors before the network.
