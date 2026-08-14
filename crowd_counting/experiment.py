"""Shared runtime and experiment helpers for the CSRNet scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def select_device(requested: str) -> torch.device:
    """Resolve a requested device and fail loudly for unavailable CUDA."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA werd expliciet gevraagd, maar PyTorch ziet geen CUDA-apparaat. "
            "Controleer nvidia-smi en installeer een CUDA-build van PyTorch."
        )
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous CUDA work before recording a duration."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed all RNGs used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    """Capture enough state to continue at the next epoch reproducibly."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any] | None, loader_generator: torch.Generator
) -> None:
    """Restore a state created by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading a checkpoint with map_location="cuda" also moves these byte
    # tensors, while the RNG setters require CPU byte tensors.
    torch.set_rng_state(state["torch"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(value: object, path: Path) -> None:
    """Write a checkpoint without exposing a partially written target file."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def summarize_count_errors(signed_errors: list[float]) -> dict[str, float | int]:
    """Calculate aggregate count-error statistics for one evaluated split."""
    if not signed_errors:
        raise ValueError("Er zijn geen voorspellingen om te evalueren.")
    errors = np.asarray(signed_errors, dtype=np.float64)
    absolute = np.abs(errors)
    return {
        "images": int(errors.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "mean_signed_error": float(errors.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "max_absolute_error": float(absolute.max()),
    }
