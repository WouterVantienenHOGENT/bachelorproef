"""Evaluate one CSRNet checkpoint on a complete image split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from csrnet import CSRNet
from data import DensityMapDataset
from experiment import select_device, summarize_count_errors, synchronize


PREDICTION_FIELDS = [
    "image",
    "image_width",
    "image_height",
    "actual_count",
    "predicted_count",
    "signed_error",
    "absolute_error",
    "squared_error",
    "model_forward_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalueer een bewaard CSRNet-checkpoint op volledige beelden."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--densities", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers moet minstens 0 zijn.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs moet minstens 0 zijn.")

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir = Path(args.output_dir).resolve()
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (predictions_path, summary_path):
        if path.exists():
            raise FileExistsError(
                f"{path} bestaat al. Kies een nieuwe --output-dir om resultaten te bewaren."
            )

    device = select_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CSRNet(pretrained_frontend=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = DensityMapDataset(args.images, args.densities, crop_size=8, mode="eval")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    first_image, _, _ = dataset[0]
    first_image = first_image.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model(first_image)
        synchronize(device)

    rows: list[dict[str, object]] = []
    signed_errors: list[float] = []
    forward_times_ms: list[float] = []
    images_root = Path(args.images).resolve()
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for images, targets, paths in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            synchronize(device)
            forward_started = time.perf_counter()
            predictions = model(images)
            synchronize(device)
            forward_ms = 1000 * (time.perf_counter() - forward_started)
            predicted_count = float(predictions.sum().item())
            actual_count = float(targets.sum().item())
            signed_error = predicted_count - actual_count
            image_path = Path(paths[0])
            relative_image = image_path.resolve().relative_to(images_root).as_posix()
            with Image.open(image_path) as source:
                width, height = source.size
            rows.append(
                {
                    "image": relative_image,
                    "image_width": width,
                    "image_height": height,
                    "actual_count": actual_count,
                    "predicted_count": predicted_count,
                    "signed_error": signed_error,
                    "absolute_error": abs(signed_error),
                    "squared_error": signed_error**2,
                    "model_forward_ms": forward_ms,
                }
            )
            signed_errors.append(signed_error)
            forward_times_ms.append(forward_ms)
    total_seconds = time.perf_counter() - evaluation_started

    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    statistics = summarize_count_errors(signed_errors)
    mean_forward_ms = float(np.mean(forward_times_ms))
    statistics.update(
        {
            "mean_actual_count": float(np.mean([row["actual_count"] for row in rows])),
            "mean_predicted_count": float(
                np.mean([row["predicted_count"] for row in rows])
            ),
            "mean_model_forward_ms": mean_forward_ms,
            "p95_model_forward_ms": float(np.percentile(forward_times_ms, 95)),
            "model_throughput_images_per_second": (
                1000 / mean_forward_ms if mean_forward_ms > 0 else None
            ),
        }
    )
    ordered_examples = sorted(rows, key=lambda row: float(row["absolute_error"]))
    examples = {
        "lowest_absolute_error": ordered_examples[0],
        "median_absolute_error": ordered_examples[len(ordered_examples) // 2],
        "highest_absolute_error": ordered_examples[-1],
    }
    summary: dict[str, object] = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "images": str(Path(args.images).resolve()),
        "densities": str(Path(args.densities).resolve()),
        "device": {
            "type": device.type,
            "name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "warmup_runs": args.warmup_runs,
        "metrics": statistics,
        "representative_examples": examples,
        "total_evaluation_seconds": total_seconds,
        "predictions_csv": str(predictions_path),
    }
    write_json(summary_path, summary)
    print(
        f"{args.split_name}: {statistics['images']} beelden | "
        f"MAE {statistics['mae']:.3f} | RMSE {statistics['rmse']:.3f} | "
        f"bias {statistics['mean_signed_error']:.3f}"
    )
    print(
        f"Model-forward: gemiddeld {mean_forward_ms:.2f} ms/beeld | "
        f"p95 {statistics['p95_model_forward_ms']:.2f} ms"
    )
    print(f"Resultaten: {output_dir}")


if __name__ == "__main__":
    main()
