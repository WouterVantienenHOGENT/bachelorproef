"""Render manual head positions and P2PNet heatmaps from saved coordinates."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]


def generate_figures():
    for scene in ("grote-markt-sint-niklaas", "trevifontein-rome"):
        assets = ROOT / "graphics/voorbeelden" / scene
        results = ROOT / "bachproef/voorbeelden/praktijk" / scene
        with Image.open(assets / "foto.png") as image:
            photo = image.convert("RGB")
        width, height = photo.size
        prediction = results / "p2pnet_shanghaitech_a"
        metadata = json.loads((prediction / "origineel_inference.json").read_text())
        with (prediction / "origineel_points.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        points = np.array([
            [float(row["x_original_px"]), float(row["y_original_px"])]
            for row in rows
        ], dtype=np.float64).reshape(-1, 2)
        assert len(points) == metadata["estimated_count"]
        points *= np.array([width, height]) / metadata["original_image_size"]
        inside = ((points >= 0) & (points < [width, height])).all(axis=1)
        assert int((~inside).sum()) == metadata["points_outside_image"]
        pixels = points[inside].astype(int)
        impulses = np.zeros((height, width), dtype=np.float64)
        np.add.at(impulses, (pixels[:, 1], pixels[:, 0]), 1)
        density = gaussian_filter(impulses, sigma=10, mode="reflect")
        assert np.isclose(density.sum(), inside.sum())
        np.save(prediction / "afgeleide_heatmap.npy", density)
        plt.imsave(assets / "p2pnet-heatmap.png", density, cmap="jet")
        (prediction / "heatmap.json").write_text(json.dumps({
            "source": "origineel_points.csv",
            "map_size": [width, height],
            "gaussian_sigma_pixels": 10,
            "boundary_mode": "reflect",
            "point_count": len(points),
            "points_outside_image": int((~inside).sum()),
            "density_sum": float(density.sum()),
            "display_normalization": "per-image minimum and maximum; jet colormap",
        }, indent=2) + "\n")

        manual = ROOT / "bachproef/voorbeelden" / scene / "handmatige-telling/markers.csv"
        if manual.exists():
            with manual.open(newline="", encoding="utf-8-sig") as stream:
                markers = list(csv.DictReader(stream))
            draw = ImageDraw.Draw(photo)
            for marker in markers:
                x = float(marker["x_normalized"]) * width
                y = float(marker["y_normalized"]) * height
                assert 0 <= x <= width and 0 <= y <= height
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="#00ffff", width=1)
            photo.save(assets / "handmatige-hoofdposities.png")
            print(f"{scene}: {len(markers)} manual markers", flush=True)
        print(f"{scene}: P2PNet heatmap sum {density.sum():.6f}", flush=True)


if __name__ == "__main__":
    generate_figures()
