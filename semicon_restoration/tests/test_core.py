"""Fast CPU tests for core restoration pipeline components."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.augmentations import SyntheticDegradation
from src.dataset import RestorationDataset, load_image
from src.losses import CompositeRestorationLoss
from src.metrics import batch_psnr, batch_ssim
from src.models.init import build_model
from src.utils import extract_state_dict, forward_tiled


class PipelineCoreTests(unittest.TestCase):
    def test_synthetic_degradation_shape_order_and_unclipped_range(self) -> None:
        image = np.linspace(0.0, 1.0, 64 * 80 * 3, dtype=np.float32).reshape(64, 80, 3)
        degradation = SyntheticDegradation(
            scale=2,
            gaussian_sigma=(0.1, 0.1),
            speckle_sigma=(0.1, 0.1),
            random_order=True,
        )
        output, metadata = degradation(image, np.random.default_rng(5))
        self.assertEqual(output.shape, (32, 40, 3))
        self.assertEqual(set(metadata["order"]), {"gaussian", "speckle", "downsample"})
        self.assertTrue(output.min() < 0.0 or output.max() > 1.0)

    def test_float_npy_range_and_paired_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_dir, noisy_dir = root / "GT", root / "NoisyLR"
            gt_dir.mkdir()
            noisy_dir.mkdir()
            gt = np.full((32, 32, 3), 0.5, dtype=np.float32)
            noisy = cv2.resize(gt, (16, 16), interpolation=cv2.INTER_CUBIC)
            noisy[0, 0] = (-0.2, 1.1, 1.4)
            cv2.imwrite(
                str(gt_dir / "sample.png"),
                cv2.cvtColor(np.rint(gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            np.save(noisy_dir / "sample.npy", noisy)

            loaded = load_image(noisy_dir / "sample.npy", channels=3)
            self.assertLess(float(loaded.min()), 0.0)
            self.assertGreater(float(loaded.max()), 1.0)
            dataset = RestorationDataset(
                gt_dir=gt_dir,
                noisy_dir=noisy_dir,
                channels=3,
                gt_patch_size=32,
                training=False,
                augment=False,
                expected_scale=2,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["input"].shape), (3, 16, 16))
            self.assertEqual(tuple(sample["target"].shape), (3, 32, 32))

    def test_synthetic_dataset_pads_gt_to_scale_multiple(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gt_dir = Path(temporary)
            odd_gt = np.full((35, 41, 3), 127, dtype=np.uint8)
            cv2.imwrite(str(gt_dir / "odd.png"), odd_gt)
            dataset = RestorationDataset(
                gt_dir=gt_dir,
                noisy_dir=None,
                channels=3,
                gt_patch_size=None,
                training=False,
                augment=False,
                synthetic_degradation=SyntheticDegradation(scale=2),
                expected_scale=2,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["target"].shape), (3, 36, 42))
            self.assertEqual(tuple(sample["input"].shape), (3, 18, 21))

    def test_nafnet_and_unet_forward_shapes(self) -> None:
        input_tensor = torch.randn(2, 3, 17, 19)
        configurations = [
            {
                "name": "nafnet",
                "img_channel": 3,
                "out_channel": 3,
                "width": 8,
                "middle_blk_num": 1,
                "enc_blk_nums": [1, 1],
                "dec_blk_nums": [1, 1],
                "scale": 2,
            },
            {
                "name": "baseline_unet",
                "img_channel": 3,
                "out_channel": 3,
                "width": 8,
                "channel_multipliers": [1, 2, 4],
                "scale": 2,
            },
        ]
        for configuration in configurations:
            with self.subTest(model=configuration["name"]):
                output = build_model(configuration)(input_tensor)
                self.assertEqual(tuple(output.shape), (2, 3, 34, 38))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_loss_backward_and_metrics(self) -> None:
        prediction = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand_like(prediction)
        criterion = CompositeRestorationLoss(lambda_lpips=0.0)
        loss, parts = criterion(prediction, target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(prediction.grad)
        self.assertEqual(set(parts), {"total", "charbonnier", "ms_ssim_loss", "lpips_loss"})
        self.assertTrue(bool(torch.isfinite(batch_psnr(prediction.detach(), target))))
        self.assertTrue(bool(torch.isfinite(batch_ssim(prediction.detach(), target))))

    def test_tiled_inference_and_checkpoint_prefix_cleanup(self) -> None:
        model = build_model(
            {
                "name": "nafnet",
                "img_channel": 1,
                "out_channel": 1,
                "width": 8,
                "middle_blk_num": 1,
                "enc_blk_nums": [1],
                "dec_blk_nums": [1],
                "scale": 3,
            }
        ).eval()
        input_tensor = torch.randn(1, 1, 27, 31)
        with torch.inference_mode():
            output = forward_tiled(model, input_tensor, scale=3, tile_size=16, overlap=4)
        self.assertEqual(tuple(output.shape), (1, 1, 81, 93))
        state = {f"module._orig_mod.{key}": value for key, value in model.state_dict().items()}
        cleaned = extract_state_dict({"model": state})
        self.assertEqual(set(cleaned), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
