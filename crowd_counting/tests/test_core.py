"""Fast regression tests for count preservation and experiment statistics."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import DensityMapDataset, resize_density_map
from experiment import atomic_torch_save, summarize_count_errors


class DensityMapTests(unittest.TestCase):
    def test_resize_preserves_total_count(self) -> None:
        density = torch.arange(35, dtype=torch.float32).reshape(5, 7)
        resized = resize_density_map(density, (3, 4))
        self.assertEqual(resized.shape, (3, 4))
        self.assertAlmostEqual(resized.sum().item(), density.sum().item(), places=4)

    def test_evaluation_pads_without_losing_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            densities = root / "density_maps"
            images.mkdir()
            densities.mkdir()
            Image.new("RGB", (13, 10), color=(100, 110, 120)).save(images / "scene.png")
            density = np.zeros((10, 13), dtype=np.float32)
            density[9, 12] = 2.5
            np.save(densities / "scene.npy", density)

            dataset = DensityMapDataset(images, densities, crop_size=8, mode="eval")
            image, target, _ = dataset[0]

            self.assertEqual(image.shape, (3, 16, 16))
            self.assertEqual(target.shape, (1, 2, 2))
            self.assertAlmostEqual(target.sum().item(), 2.5, places=5)


class ExperimentTests(unittest.TestCase):
    def test_count_error_statistics(self) -> None:
        statistics = summarize_count_errors([-2.0, 0.0, 4.0])
        self.assertEqual(statistics["images"], 3)
        self.assertAlmostEqual(statistics["mae"], 2.0)
        self.assertAlmostEqual(statistics["rmse"], math.sqrt(20 / 3))
        self.assertAlmostEqual(statistics["mean_signed_error"], 2 / 3)
        self.assertAlmostEqual(statistics["median_absolute_error"], 2.0)
        self.assertAlmostEqual(statistics["max_absolute_error"], 4.0)

    def test_atomic_checkpoint_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "last.pt"
            atomic_torch_save({"epoch": 3}, checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertFalse(checkpoint_path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
