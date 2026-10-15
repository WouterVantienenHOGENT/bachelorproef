"""Shared CSRNet and P2PNet inference for CLI and Streamlit."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from csrnet import CSRNet
from data import IMAGENET_MEAN, IMAGENET_STD
from experiment import select_device, synchronize


def describe_device(device: torch.device) -> str:
    """Return a useful hardware label for benchmark output."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        fields = {
            key.strip(): value.strip()
            for line in cpuinfo.splitlines()
            if ":" in line
            for key, value in [line.split(":", maxsplit=1)]
        }
        if fields.get("Model"):
            return fields["Model"]
        if fields.get("model name"):
            return fields["model name"]
    except OSError:
        pass
    return platform.processor().strip() or platform.machine() or "CPU"


@dataclass
class InferenceOptions:
    checkpoint: str
    image: str
    output_dir: str = "predictions"
    device: str = "auto"
    warmup_runs: int = 0
    timed_runs: int = 1
    benchmark_csv: str = "runs/inference_benchmarks.csv"
    architecture: str = "auto"
    max_edge: int = 0
    threshold: float = 0.5


@dataclass
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    epoch: int | None
    architecture: str = "csrnet"


def load_model(checkpoint_path: str | Path, requested_device: str = "auto",
               architecture: str = "auto") -> LoadedModel:
    device = select_device(requested_device)
    # Training checkpoints also contain optimizer state. Keep that off the GPU.
    if architecture not in {"auto", "csrnet", "p2pnet"}:
        raise ValueError(f"Onbekende architectuur: {architecture}")
    # Compact catalog weights and official P2PNet work with weights_only=True.
    # Legacy CSRNet training checkpoints include NumPy RNG state and are local trusted files.
    if architecture == "p2pnet" or Path(checkpoint_path).suffix.lower() == ".pth":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if architecture == "auto":
        if "backbone.body1.0.weight" in state and "classification.output.weight" in state:
            architecture = "p2pnet"
        elif "frontend.0.weight" in state and "output_layer.weight" in state:
            architecture = "csrnet"
        else:
            raise ValueError("Checkpoint is geen ondersteund CSRNet- of P2PNet-model.")
    if architecture == "p2pnet":
        from vendor.p2pnet.backbone import build_backbone
        from vendor.p2pnet.network import P2PNet
        model = P2PNet(build_backbone())
    else:
        model = CSRNet(pretrained_frontend=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    synchronize(device)
    return LoadedModel(model, device, Path(checkpoint_path).resolve(), checkpoint.get("epoch"), architecture)


def prepare_image(image, architecture, max_edge):
    """Limit image size, then apply the architecture's documented sizing rule."""
    if max_edge and max(image.size) > max_edge:
        ratio = max_edge / max(image.size)
        image = image.resize(tuple(max(1, round(side * ratio)) for side in image.size), Image.Resampling.LANCZOS)
    if architecture == "p2pnet":
        # Official run_test.py resizes each dimension down to a multiple of 128.
        # Small images use at least 128 pixels rather than the original zero-size error.
        image = image.resize(tuple(max(128, side // 128 * 128) for side in image.size), Image.Resampling.LANCZOS)
    return image


def select_points(prediction, threshold, input_size, original_size):
    """Match official score > threshold counting; retain even out-of-frame proposals."""
    scores = prediction["pred_logits"][0].softmax(-1)[:, 1]
    keep = scores > threshold
    points = prediction["pred_points"][0, keep].cpu().numpy().copy()
    points *= np.asarray(original_size) / np.asarray(input_size)
    return points, scores[keep].cpu().numpy()


def predict_image(args: InferenceOptions, loaded: LoadedModel | None = None) -> dict:
    """Run the same preprocessing, inference and exports for CLI and GUI."""
    if args.warmup_runs < 0 or args.timed_runs < 1:
        raise ValueError("--warmup-runs moet >= 0 en --timed-runs moet >= 1 zijn.")
    if args.max_edge < 0 or not 0 <= args.threshold <= 1:
        raise ValueError("max_edge moet >= 0 en threshold moet tussen 0 en 1 liggen.")
    total_started = time.perf_counter()
    reused = loaded is not None
    if loaded is None:
        loaded = load_model(args.checkpoint, args.device, args.architecture)
    elif (loaded.checkpoint_path != Path(args.checkpoint).resolve()
          or loaded.device != select_device(args.device)
          or args.architecture not in {"auto", loaded.architecture}):
        raise ValueError("Geladen model komt niet overeen met checkpoint of rekenapparaat.")
    model, device = loaded.model, loaded.device
    load_ms = 1000 * (time.perf_counter() - total_started)
    prepare_started = time.perf_counter()
    with Image.open(args.image) as source:
        original = source.convert("RGB")
    image = prepare_image(original, loaded.architecture, args.max_edge)
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255
    tensor = (
        tensor - torch.tensor(IMAGENET_MEAN)[:, None, None]
    ) / torch.tensor(IMAGENET_STD)[:, None, None]
    # Padding lets arbitrary image sizes pass through the stride-8 model.
    pad_h, pad_w = (-image.height) % 8, (-image.width) % 8
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h)).unsqueeze(0).to(device)
    synchronize(device)
    prepare_ms = 1000 * (time.perf_counter() - prepare_started)
    with torch.inference_mode():
        warmup_started = time.perf_counter()
        for _ in range(args.warmup_runs):
            model(tensor)
        synchronize(device)
        warmup_ms = 1000 * (time.perf_counter() - warmup_started)
        forward_started = time.perf_counter()
        prediction = None
        for _ in range(args.timed_runs):
            prediction = model(tensor)
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        assert prediction is not None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    if loaded.architecture == "csrnet":
        density = prediction[0, 0].cpu().numpy()
        density = density[: (image.height + 7) // 8, : (image.width + 7) // 8]
        np.save(output_dir / f"{stem}_density.npy", density)
        plt.imsave(output_dir / f"{stem}_density.png", density, cmap="jet")
        count = float(density.sum())
        exports = [f"{stem}_density.png", f"{stem}_density.npy"]
        extra = {"prediction_type": "density", "visualization_caption": "Voorspelde density map"}
    else:
        points, scores = select_points(prediction, args.threshold, image.size, original.size)
        count = len(points)
        with (output_dir / f"{stem}_points.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x_original_px", "y_original_px", "confidence"])
            writer.writerows((float(x), float(y), float(score)) for (x, y), score in zip(points, scores))
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        radius = max(2, round(max(original.size) / 650))
        for x, y in points:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="#00ffff", width=max(1, radius // 3))
        overlay.save(output_dir / f"{stem}_points.png")
        exports = [f"{stem}_points.png", f"{stem}_points.csv"]
        outside = ((points < 0).any(axis=1) | (points >= np.asarray(original.size)).any(axis=1)).sum()
        extra = {"prediction_type": "points", "threshold": args.threshold,
                 "points_coordinate_system": "original image pixels", "points_outside_image": int(outside),
                 "visualization_caption": "Voorspelde hoofdposities"}
    forward_ms = 1000 * forward_seconds / args.timed_runs
    frames_per_second = 1000 / forward_ms
    total_ms = 1000 * (time.perf_counter() - total_started)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": loaded.epoch,
        "architecture": loaded.architecture,
        "image": str(Path(args.image).resolve()),
        "image_size": [image.width, image.height],
        "original_image_size": list(original.size),
        "max_edge": args.max_edge,
        "preprocessing": "ImageNet normalization; " + (
            "LANCZOS resize down to multiples of 128 (minimum 128)" if loaded.architecture == "p2pnet"
            else "aspect-ratio resize if limited; normalized zero padding to multiple of 8"),
        "estimated_count": count,
        "device": str(device),
        "device_name": describe_device(device),
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "mean_forward_ms": forward_ms,
        "model_throughput_fps": frames_per_second,
        "total_command_ms": total_ms,
        "model_reused": reused,
        "model_load_ms": load_ms,
        "preprocess_ms": prepare_ms,
        "warmup_ms": warmup_ms,
        "exports": exports + [f"{stem}_inference.json"],
        **extra,
    }
    (output_dir / f"{stem}_inference.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.benchmark_csv:
        benchmark_path = Path(args.benchmark_csv)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_fields = [
            "timestamp_utc",
            "run",
            "checkpoint_epoch",
            "device",
            "device_name",
            "image",
            "image_width",
            "image_height",
            "estimated_count",
            "warmup_runs",
            "timed_runs",
            "mean_forward_ms",
            "model_throughput_fps",
            "total_command_ms",
        ]
        benchmark_row = {
            "timestamp_utc": result["timestamp_utc"],
            "run": Path(args.checkpoint).stem if Path(args.checkpoint).parent.name in {"csrnet", "p2pnet"} else Path(args.checkpoint).parent.name,
            "checkpoint_epoch": result["checkpoint_epoch"],
            "device": result["device"],
            "device_name": result["device_name"],
            "image": Path(args.image).name,
            "image_width": image.width,
            "image_height": image.height,
            "estimated_count": result["estimated_count"],
            "warmup_runs": result["warmup_runs"],
            "timed_runs": result["timed_runs"],
            "mean_forward_ms": result["mean_forward_ms"],
            "model_throughput_fps": result["model_throughput_fps"],
            "total_command_ms": result["total_command_ms"],
        }
        write_header = not benchmark_path.exists()
        with benchmark_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=benchmark_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(benchmark_row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--timed-runs", type=int, default=1)
    parser.add_argument("--architecture", choices=("auto", "csrnet", "p2pnet"), default="auto")
    parser.add_argument("--max-edge", type=int, default=0, help="Langste beeldzijde voor inferentie; 0 behoudt de invoerresolutie.")
    parser.add_argument("--threshold", type=float, default=0.5, help="P2PNet: tel punten met een score boven deze grens.")
    parser.add_argument(
        "--benchmark-csv",
        default="runs/inference_benchmarks.csv",
        help=(
            "Centrale CSV voor vergelijkbare inferentiemetingen; "
            "leeg om uit te schakelen."
        ),
    )
    args = parser.parse_args()
    result = predict_image(InferenceOptions(**vars(args)))
    print(f"Geschat aantal personen: {result['estimated_count']:.2f}")
    print(
        f"Inference ({result['device']}): gemiddeld {result['mean_forward_ms']:.2f} ms per beeld "
        f"of {result['model_throughput_fps']:.3f} FPS over {args.timed_runs} runs"
    )


if __name__ == "__main__":
    main()
