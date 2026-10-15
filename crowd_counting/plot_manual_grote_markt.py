"""Plot the exported manual annotations; no model inference is performed."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bachproef/voorbeelden/grote-markt-sint-niklaas"
DATA = RESULTS / "handmatige-telling"
ASSETS = ROOT / "graphics/voorbeelden/grote-markt-sint-niklaas"


def main():
    with (DATA / "markers.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with (DATA / "summary.csv").open(newline="", encoding="utf-8-sig") as stream:
        totals = [r for r in csv.DictReader(stream)
                  if r["scope"] == "image" and r["category"] == "Total"]
    assert len(totals) == 1 and int(totals[0]["count"]) == len(rows) == 738
    assert len({r["marker_id"] for r in rows}) == len(rows)
    assert all(r["marker_shape"] == "point" and r["category"] == "People" for r in rows)
    with Image.open(ASSETS / "origineel.png") as source:
        assert all((int(r["image_width"]), int(r["image_height"])) == source.size for r in rows)
        width = 1024
        height = round(source.height * width / source.width)
        photo = np.asarray(source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))
    points = np.array([[float(r["x_normalized"]), float(r["y_normalized"])] for r in rows])
    assert np.all((points >= 0) & (points <= 1))
    points *= [width, height]
    impulses = np.zeros((height, width), dtype=np.float64)
    pixels = np.minimum(points.astype(int), [width - 1, height - 1])
    np.add.at(impulses, (pixels[:, 1], pixels[:, 0]), 1)
    density = gaussian_filter(impulses, sigma=10, mode="reflect")
    assert np.isclose(density.sum(), len(rows))
    relative = density / density.max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].imshow(photo, extent=(0, width, height, 0))
    axes[0].scatter(points[:, 0], points[:, 1], s=5, facecolors="none", edgecolors="#00ffff", linewidths=0.45)
    axes[0].set_title("738 handmatig aangeduide hoofden", fontsize=12)
    axes[1].imshow(photo, extent=(0, width, height, 0))
    axes[1].imshow(relative, extent=(0, width, height, 0), cmap="inferno", vmin=0, vmax=1,
                   alpha=np.where(relative > 0.015, 0.78, 0))
    axes[1].set_title("Spreiding van de annotaties", fontsize=12)
    for ax in axes:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="inferno"),
                       ax=axes[1], orientation="horizontal", fraction=0.045, pad=0.025)
    bar.set_label("Relatieve annotatiedichtheid (maximum = 1)", fontsize=10)
    fig.savefig(ASSETS / "handmatige-telling.png", dpi=220)
    plt.close(fig)
    comparison = json.loads((RESULTS / "comparison.json").read_text())
    metrics = [{"label": r["label"], "estimated_count": r["estimated_count"],
                "shortfall_vs_annotations": len(rows) - r["estimated_count"],
                "shortfall_percent": 100 * (len(rows) - r["estimated_count"]) / len(rows)}
               for r in comparison["results"]]
    (DATA / "analyse.json").write_text(json.dumps({
        "annotation_count": len(rows), "complete_reference": False,
        "map_size": [width, height], "gaussian_sigma_pixels": 10,
        "boundary_mode": "reflect", "density_sum": float(density.sum()),
        "display_normalization": "divide by maximum; independent of model heatmap",
        "models": metrics,
    }, indent=2) + "\n")
    print(f"Verified {len(rows)} markers; density sum = {density.sum():.6f}")


if __name__ == "__main__":
    main()
