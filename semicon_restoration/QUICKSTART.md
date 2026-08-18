# Quick Start for the Downloaded Workspace

## What is already included

- Fine-tuned model: `weights/real_kla/best_model.pth`
- Restored 400-image test archive: `results/real_kla_restored_test.zip`
- Training/inference source code
- Fine-tuning config: `configs/real_kla_finetune.yaml`
- Validation split manifest: `results/real_kla_val_ids.txt`
- Training report and metrics

The original extracted 1.1 GB training corpus may not be included in a workspace download because generated `out/` directories are snapshot-excluded. It is not required for inference.

## 1. Create the environment

Run commands from the `semicon_restoration` directory.

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### NVIDIA GPU installation

Use the PyTorch index compatible with your NVIDIA driver. CUDA 12.8 example:

```bash
  pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### CPU-only installation

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Confirm the environment:

```bash
python -m unittest discover -s tests -v
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Restore new images

### Restore one file

Copy the degraded `.npy` or `.png` file into the project, for example `my_image.npy`, then run:

```bash
python restore_image.py \
  --input_file my_image.npy \
  --output_file my_image_restored.png \
  --weights weights/real_kla/best_model.pth \
  --device cpu
```

Use `--device cuda` on an NVIDIA GPU. If `--output_file` is omitted, the result is written to `restored/<name>_restored.png`.

### Restore a folder/batch

Create an input directory:

```text
my_inputs/
├── image_001.npy
└── image_002.npy
```

Recommended input format:

- float32 NPY
- HxW grayscale array
- 128x128 for the same distribution as the supplied test set
- Values may remain outside `[0,1]`; do not clamp them

PNG files are also accepted. The checkpoint is grayscale and produces a result at exactly twice the input height and width.

### NVIDIA GPU

```bash
python inference.py \
  --input_dir my_inputs \
  --output_dir my_restored \
  --weights weights/real_kla/best_model.pth \
  --batch_size 8 \
  --device cuda \
  --save_format png \
  --bit_depth 16
```

### CPU

```bash
python inference.py \
  --input_dir my_inputs \
  --output_dir my_restored \
  --weights weights/real_kla/best_model.pth \
  --batch_size 4 \
  --device cpu \
  --num_workers 0 \
  --save_format png \
  --bit_depth 16
```

Outputs are clamped to `[0,1]` and saved at x2 resolution. Use `--save_format npy` when floating-point outputs are required.

For many same-sized images on an NVIDIA GPU, optionally add `--compile`. The first call will be slower because compilation occurs once.

## 3. Use the already restored test submission

No inference is needed if you only need the 400 restored test files. Extract:

```text
results/real_kla_restored_test.zip
```

Archive SHA-256:

```text
14e980589685afb23a87a82702930b020b41e77fb207c2fd35a0e079f106198c
```

## 4. Continue training or fine-tune

Training requires the paired dataset to exist at the paths in `configs/real_kla_finetune.yaml`:

```text
out/real_data/train/GT/
out/real_data/train/NoisyLR/
out/real_data/val/GT/
out/real_data/val/NoisyLR/
```

Each GT/NoisyLR pair must use the same stem. For the supplied data, GT is 256x256 and NoisyLR is 128x128.

Start a fresh fine-tuning optimizer from the current best model:

```bash
python train.py \
  --config configs/real_kla_finetune.yaml \
  --pretrained weights/real_kla/best_model.pth \
  --device cuda
```

During a new training run, `checkpoints/last.pth` is generated. Resume that future run with `--resume <path-to-last.pth>`. Use `--pretrained` and `--resume` mutually exclusively.

## Important limitations

- The provided checkpoint is a compact grayscale x2 model.
- It was selected using the supplied data distribution.
- It will always double spatial dimensions.
- It cannot reliably invent details absent from the LR image.
- Keep an untouched validation set when doing additional training.
