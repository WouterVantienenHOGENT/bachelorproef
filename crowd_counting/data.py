"""Dataset utilities for image/density-map pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_density_map(density: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a density map while preserving its total count."""
    original_sum = density.sum()
    if original_sum.abs() < 1e-12:
        return density.new_zeros(size)
    is_downsample = size[0] <= density.shape[0] and size[1] <= density.shape[1]
    if is_downsample:
        # Area interpolation accounts for every source pixel. Bilinear
        # downsampling can completely miss a narrow peak near an image edge.
        resized = F.interpolate(density[None, None], size=size, mode="area")[0, 0]
    else:
        resized = F.interpolate(
            density[None, None], size=size, mode="bilinear", align_corners=False
        )[0, 0]
    resized_sum = resized.sum()
    if resized_sum.abs() < 1e-12:
        raise ValueError("Density map verloor haar volledige massa bij het schalen.")
    # Interpolation averages values, so explicitly restore the integral.
    return resized * (original_sum / resized_sum)


class DensityMapDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Pairs ``images/foo.jpg`` with ``density_maps/foo.npy`` recursively."""

    def __init__(
        self,
        images_dir: str | Path,
        densities_dir: str | Path,
        crop_size: int = 512,
        mode: Literal["train", "eval", "val"] = "train",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.densities_dir = Path(densities_dir)
        self.crop_size = crop_size
        self.mode = mode
        self.samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.images_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(self.images_dir).with_suffix(".npy")
            density_path = self.densities_dir / relative
            if density_path.exists():
                self.samples.append((image_path, density_path))
        if not self.samples:
            raise ValueError(
                "Geen beeld/density-map-paren gevonden in "
                f"{self.images_dir} en {self.densities_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(
        self, image: Image.Image, density: torch.Tensor
    ) -> tuple[Image.Image, torch.Tensor]:
        width, height = image.size
        if min(width, height) < self.crop_size:
            scale = self.crop_size / min(width, height)
            new_size = (round(width * scale), round(height * scale))
            image = image.resize(new_size, Image.Resampling.BILINEAR)
            density = resize_density_map(density, (new_size[1], new_size[0]))
            width, height = new_size
        if self.mode == "train":
            left = torch.randint(0, width - self.crop_size + 1, ()).item()
            top = torch.randint(0, height - self.crop_size + 1, ()).item()
        else:
            left, top = (width - self.crop_size) // 2, (height - self.crop_size) // 2
        return (
            image.crop((left, top, left + self.crop_size, top + self.crop_size)),
            density[top : top + self.crop_size, left : left + self.crop_size],
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, density_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        density = torch.from_numpy(np.load(density_path).astype(np.float32))
        if density.ndim != 2 or density.shape != (image.height, image.width):
            raise ValueError(f"{density_path} moet een 2D-map met beeldafmetingen bevatten.")
        if self.mode == "train":
            image, density = self._crop(image, density)
        image_tensor = (
            torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        )
        image_tensor = (
            image_tensor - torch.tensor(IMAGENET_MEAN)[:, None, None]
        ) / torch.tensor(IMAGENET_STD)[:, None, None]
        if self.mode != "train":
            # Preserve every annotation during evaluation. Padding, unlike
            # cropping, does not discard people near the right or bottom edge.
            pad_h = (-image.height) % 8
            pad_w = (-image.width) % 8
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
            density = F.pad(density, (0, pad_w, 0, pad_h))
        # CSRNet has output stride 8; preserve density-map count at that resolution.
        output_size = (image_tensor.shape[-2] // 8, image_tensor.shape[-1] // 8)
        target = resize_density_map(density, output_size)
        return image_tensor, target.unsqueeze(0), str(image_path)
