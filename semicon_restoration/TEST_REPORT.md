# Verification Report

**Date:** 2026-08-16  
**Test host:** Linux, Python 3.13.14, PyTorch 2.11.0+cpu, torchvision 0.26.0+cpu

## Result

All executable CPU checks passed. No NVIDIA GPU is exposed in the test sandbox, so CUDA AMP, GPU `torch.compile`, and real GPU throughput still need to be benchmarked on the target NVIDIA machine. The corresponding code paths use supported PyTorch 2.11 APIs.

## Automated core suite

```bash
python -m unittest discover -s tests -v
```

Result: **6/6 passed**.

Coverage:

- Gaussian, speckle, and downsampling application and permutation
- Preservation of floating NoisyLR values outside `[0, 1]`
- Strict paired dataset loading
- GT-only synthetic dataset and scale-divisible border padding
- NAFNet and baseline U-Net forward shapes on odd image dimensions
- Composite loss backward pass
- PSNR and SSIM computation
- Overlap-tiled inference
- Compiled/raw checkpoint prefix cleanup

## End-to-end training check

A temporary paired dataset was generated with three training images and two validation images. Inputs were float32 NPY files containing out-of-range values; targets were PNG files. A one-epoch CLI run completed successfully:

```text
Device=cpu | train=3 | val=2 | parameters=20,940 | AMP=False
Epoch 001/1: train=0.23091 PSNR=14.969 SSIM=0.28121
```

Verified artifacts:

- `last.pth`
- `epoch_0001.pth`
- `best_model.pth`
- `best_psnr.pth`
- `best_ssim.pth`
- mirrored `weights/best_model.pth`
- `history.csv`
- resolved YAML config
- TensorBoard event file

A separate run with LPIPS enabled completed successfully and reported PSNR, SSIM, and LPIPS. A three-microbatch run with `accumulation_steps: 2` also passed, including the final partial accumulation group.

## Default-configuration check

The unmodified default NAFNet configuration was instantiated and tested:

```text
Parameters: 14,353,228
Input:      1 x 3 x 32 x 32
Output:     1 x 3 x 64 x 64
Loss:       Charbonnier + MS-SSIM + pretrained AlexNet LPIPS
Backward:   all gradients finite
```

The pretrained AlexNet LPIPS backbone downloaded and executed successfully. A grayscale x3 model and loss pass also succeeded.

## Inference checks

The standalone CLI was run against equal-sized and mixed-sized input images.

Passed modes:

- Full-frame inference
- Overlap-tiled inference
- Shape grouping inside a batch
- `num_workers=0` and `num_workers=2`
- Asynchronous image writing
- float32 NPY output
- 8-bit PNG output
- 16-bit PNG output
- Nested input-directory reproduction
- Embedded checkpoint configuration loading
- Final output range strictly `[0, 1]`

Verified output dimensions included `32x32 -> 64x64` and odd `35x29 -> 70x58` at scale x2.

## Compile and environment checks

- `torch.compile` forward pass: passed
- Python compile/AST check: passed
- `train.py --help`: passed
- `inference.py --help`: passed
- YAML scale consistency: passed
- `pip check`: no broken requirements
- Pinned requirements dry-run resolution: passed

## Corrections made during verification

1. Fixed deeper baseline U-Net encoder channel wiring.
2. Added scale-divisible padding for uncropped synthetic GT images.
3. Added upfront rejection of synthetic scales incompatible with the fixed model scale.
4. Corrected scaling for a final partial gradient-accumulation group.
5. Reused the loss LPIPS result during validation, avoiding a second AlexNet forward per batch.
6. Shared the LPIPS module between loss and metrics, reducing validation memory.
7. Updated pins to a Python 3.10–3.13-compatible PyTorch 2.11 stack.
8. Added the reproducible six-test CPU suite.
9. Improved disabled/non-validation metric display from `nan` to `N/A`.

## Target-GPU acceptance test

On the actual NVIDIA deployment host, run:

```bash
python -m unittest discover -s tests -v
python train.py --config configs/train_config.yaml --device cuda --epochs 1
python inference.py \
  --input_dir <validation_noisylr> \
  --output_dir <temporary_output> \
  --weights weights/best_model.pth \
  --batch_size 8 --device cuda --compile
```

Tune `batch_size`, `num_workers`, `io_workers`, `tile_size`, AMP, and compile mode using representative challenge images. GPU throughput cannot be inferred reliably from this CPU-only sandbox.
