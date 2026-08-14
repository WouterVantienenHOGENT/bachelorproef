"""Generate and export a CSRNet density map for one image."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from csrnet import CSRNet
from data import IMAGENET_MEAN, IMAGENET_STD
from experiment import select_device, synchronize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument(
        "--benchmark-csv",
        default="runs/inference_benchmarks.csv",
        help=(
            "Centrale CSV voor vergelijkbare inferentiemetingen; "
            "leeg om uit te schakelen."
        ),
    )
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.timed_runs < 1:
        raise ValueError("--warmup-runs moet >= 0 en --timed-runs moet >= 1 zijn.")
    total_started = time.perf_counter()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CSRNet(pretrained_frontend=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255
    tensor = (
        tensor - torch.tensor(IMAGENET_MEAN)[:, None, None]
    ) / torch.tensor(IMAGENET_STD)[:, None, None]
    # Padding lets arbitrary image sizes pass through the stride-8 model.
    pad_h, pad_w = (-image.height) % 8, (-image.width) % 8
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h)).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model(tensor)
        synchronize(device)
        forward_started = time.perf_counter()
        prediction = None
        for _ in range(args.timed_runs):
            prediction = model(tensor)
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        assert prediction is not None
        density = prediction[0, 0].cpu().numpy()
    density = density[: (image.height + 7) // 8, : (image.width + 7) // 8]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    np.save(output_dir / f"{stem}_density.npy", density)
    plt.imsave(output_dir / f"{stem}_density.png", density, cmap="jet")
    forward_ms = 1000 * forward_seconds / args.timed_runs
    total_ms = 1000 * (time.perf_counter() - total_started)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "image": str(Path(args.image).resolve()),
        "image_size": [image.width, image.height],
        "estimated_count": float(density.sum()),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "mean_forward_ms": forward_ms,
        "total_command_ms": total_ms,
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
            "total_command_ms",
        ]
        benchmark_row = {
            "timestamp_utc": result["timestamp_utc"],
            "run": Path(args.checkpoint).parent.name,
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
            "total_command_ms": result["total_command_ms"],
        }
        write_header = not benchmark_path.exists()
        with benchmark_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=benchmark_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(benchmark_row)
    print(f"Geschat aantal personen: {density.sum():.2f}")
    print(
        f"Inference ({device}): gemiddeld {forward_ms:.2f} ms per beeld "
        f"over {args.timed_runs} runs"
    )


if __name__ == "__main__":
    main()
