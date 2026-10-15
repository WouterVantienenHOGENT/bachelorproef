"""Verify model discovery, architecture isolation and exported point coordinates."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from infer import InferenceOptions, LoadedModel, predict_image, prepare_image
from model_catalog import discover_models


class CatalogTests(unittest.TestCase):
    def test_copied_weights_names_metadata_and_invalid_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "p2pnet").mkdir()
            copied = root / "p2pnet/Feest_model.pth"
            copied.touch()
            entries, errors = discover_models(root)
            self.assertFalse(errors)
            self.assertEqual((entries[0].name, entries[0].architecture), ("Feest model", "p2pnet"))
            copied.with_suffix(".json").write_text(json.dumps({"name": "Mijn voorgetraind model"}))
            unknown = root / "unknown.pt"
            unknown.touch()
            entries, errors = discover_models(root)
            self.assertEqual(entries[0].name, "Mijn voorgetraind model")
            self.assertEqual(len(errors), 1)
            copied.with_suffix(".json").write_text("[]")
            entries, errors = discover_models(root)
            self.assertEqual(entries, [])
            self.assertEqual(len(errors), 2)


class PointModel(torch.nn.Module):
    def forward(self, image):
        return {
            # First and third are accepted; equality at 0.5 is not accepted.
            "pred_logits": torch.tensor([[[0., 2.], [0., 0.], [0., 3.]]], device=image.device),
            "pred_points": torch.tensor([[[64., 64.], [0., 0.], [-1., 10.]]], device=image.device),
        }


class InferenceTests(unittest.TestCase):
    def test_architecture_specific_sizing(self):
        image = Image.new("RGB", (5120, 3415))
        self.assertEqual(prepare_image(image, "csrnet", 1024).size, (1024, 683))
        self.assertEqual(prepare_image(image, "p2pnet", 1024).size, (1024, 640))
        self.assertEqual(prepare_image(Image.new("RGB", (12, 20)), "p2pnet", 0).size, (128, 128))
        self.assertEqual(prepare_image(image, "csrnet", 0).size, image.size)

    def test_point_export_threshold_original_coordinates_and_reuse_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (256, 256)).save(image)
            checkpoint = root / "model.pth"
            loaded = LoadedModel(PointModel(), torch.device("cpu"), checkpoint.resolve(), None, "p2pnet")
            args = InferenceOptions(str(checkpoint), str(image), str(root / "out"),
                                    device="cpu", architecture="p2pnet", max_edge=128, benchmark_csv="")
            result = predict_image(args, loaded)
            self.assertEqual(result["estimated_count"], 2)
            self.assertEqual(result["points_outside_image"], 1)
            self.assertEqual(result["image_size"], [128, 128])
            self.assertEqual(result["original_image_size"], [256, 256])
            with (root / "out/input_points.csv").open() as stream:
                points = list(csv.DictReader(stream))
            self.assertEqual(float(points[0]["x_original_px"]), 128)
            self.assertEqual(float(points[0]["y_original_px"]), 128)
            self.assertEqual(float(points[1]["x_original_px"]), -2)
            self.assertEqual(set(result["exports"]), {p.name for p in (root / "out").iterdir()})
            args.architecture = "csrnet"
            with self.assertRaisesRegex(ValueError, "Geladen model"):
                predict_image(args, loaded)

    def test_p2pnet_cpu_anchors_do_not_follow_global_cuda(self):
        from vendor.p2pnet.network import AnchorPoints
        with patch("torch.cuda.is_available", return_value=True):
            points = AnchorPoints(pyramid_levels=[3], row=2, line=2)(torch.zeros(1, 3, 128, 128))
        self.assertEqual(points.device.type, "cpu")
        self.assertEqual(points.shape, (1, 1024, 2))
        np.testing.assert_array_equal(points[0, :4].numpy(), [[2, 2], [6, 2], [2, 6], [6, 6]])


if __name__ == "__main__":
    unittest.main()
