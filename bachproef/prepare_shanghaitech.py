"""Prepare ShanghaiTech Part B for reproducible CSRNet experiments.

The script keeps the original test split untouched, and makes a deterministic
validation split from the official training set.  It converts the MATLAB head
point annotations into density maps whose sum equals the ground-truth count.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory containing ShanghaiTech part_B.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Fixed Gaussian sigma in pixels for Part B.")
    return parser.parse_args()


def load_points(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    image_info = data["image_info"]
    points = image_info[0, 0][0, 0][0]
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def make_density_map(points: np.ndarray, height: int, width: int, sigma: float) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        col = int(np.clip(round(x), 0, width - 1))
        row = int(np.clip(round(y), 0, height - 1))
        density[row, col] += 1.0
    density = gaussian_filter(density, sigma=sigma, mode="constant")
    total = density.sum()
    if total > 0:
        density *= len(points) / total
    return density


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_existing_names(target: Path, names: list[str]) -> None:
    expected_images = set(names)
    expected_densities = {str(Path(name).with_suffix(".npy")) for name in names}
    existing_images = {
        path.name for path in (target / "images").glob("*") if path.is_file()
    }
    existing_densities = {
        path.name for path in (target / "density_maps").glob("*") if path.is_file()
    }
    unexpected_images = existing_images - expected_images
    unexpected_densities = existing_densities - expected_densities
    if unexpected_images or unexpected_densities:
        raise ValueError(
            f"{target} bevat bestanden uit een andere splitsing. "
            "Gebruik een nieuwe --output-map om datasets niet te vermengen."
        )


def prepare_split(source: Path, target: Path, names: list[str], sigma: float) -> None:
    images_dir = source / "images"
    ground_truth_dir = source / "ground-truth"
    validate_existing_names(target, names)
    for name in names:
        image_path = images_dir / name
        density_path = target / "density_maps" / Path(name).with_suffix(".npy")
        image_target = target / "images" / name
        density_path.parent.mkdir(parents=True, exist_ok=True)
        image_target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            width, height = image.size
        points = load_points(ground_truth_dir / f"GT_{Path(name).stem}.mat")
        if image_target.is_file() and density_path.is_file():
            existing_density = np.load(density_path, mmap_mode="r")
            if existing_density.shape != (height, width) or not np.isclose(
                existing_density.sum(), len(points), rtol=0, atol=1e-3
            ):
                raise ValueError(f"Ongeldige bestaande density map: {density_path}")
            continue
        density = make_density_map(points, height, width, sigma)
        if not np.isclose(density.sum(), len(points), rtol=0, atol=1e-3):
            raise ValueError(f"Count mismatch for {image_path}")
        shutil.copy2(image_path, image_target)
        np.save(density_path, density)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if args.sigma <= 0:
        raise ValueError("--sigma must be greater than 0.")
    official_train = args.source / "train_data"
    official_test = args.source / "test_data"
    if not (official_train / "images").is_dir() or not (official_test / "images").is_dir():
        raise FileNotFoundError("Expected train_data and test_data in --source.")
    names = sorted(path.name for path in (official_train / "images").glob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    validation_indices = set(
        rng.choice(
            len(names), size=round(len(names) * args.val_fraction), replace=False
        )
    )
    validation_names = [name for index, name in enumerate(names) if index in validation_indices]
    training_names = [name for index, name in enumerate(names) if index not in validation_indices]
    test_names = sorted(path.name for path in (official_test / "images").glob("*.jpg"))
    if not names or not test_names:
        raise ValueError("No images found.")
    manifest: dict[str, object] = {
        "format_version": 1,
        "dataset": "ShanghaiTech Part B",
        "validation_fraction": args.val_fraction,
        "split_seed": args.seed,
        "gaussian_sigma_pixels": args.sigma,
        "splits": {
            "train": training_names,
            "validation": validation_names,
            "test": test_names,
        },
    }
    manifest_path = args.output / "dataset_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                f"{manifest_path} beschrijft andere instellingen of splits. "
                "Gebruik een nieuwe --output-map."
            )
    prepare_split(official_train, args.output / "train", training_names, args.sigma)
    prepare_split(official_train, args.output / "val", validation_names, args.sigma)
    prepare_split(official_test, args.output / "test", test_names, args.sigma)
    write_json(manifest_path, manifest)
    print(
        f"Prepared {len(training_names)} train, {len(validation_names)} val, "
        f"{len(test_names)} test images. Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
