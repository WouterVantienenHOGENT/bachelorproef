"""Fine-tune CSRNet on image/density-map pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from csrnet import CSRNet
from data import DensityMapDataset
from experiment import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    select_device,
    summarize_count_errors,
    synchronize,
)


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "val_mae",
    "val_rmse",
    "train_seconds",
    "validation_seconds",
    "epoch_seconds",
    "val_forward_ms_per_image",
    "peak_cuda_memory_mib",
    "completed_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet voor crowd counting.")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-densities", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-densities", required=True)
    parser.add_argument(
        "--output-dir",
        help="Nieuwe runmap; bij --resume standaard de map van het checkpoint.",
    )
    parser.add_argument(
        "--resume",
        help="Hervat een onderbroken run vanaf last.pt. --epochs blijft het totale doel.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512, help="Veelvoud van 8.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Bewaar naast best.pt en last.pt iedere N epochs een snapshot; 0 schakelt dit uit.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Gebruik 'cuda' om niet onopgemerkt op CPU te trainen.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Vraag deterministische PyTorch-algoritmen voor reproduceerbare eindruns.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def last_logged_epoch(metrics_path: Path) -> int:
    if not metrics_path.is_file():
        return 0
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return int(rows[-1]["epoch"]) if rows else 0


def append_metric(metrics_path: Path, metric: dict[str, object]) -> None:
    with metrics_path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writerow(metric)


def dataset_manifest_metadata(
    train_images: str, val_images: str
) -> dict[str, object] | None:
    train_root = Path(train_images).resolve().parents[1]
    val_root = Path(val_images).resolve().parents[1]
    if train_root != val_root:
        return None
    manifest_path = train_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None
    content = manifest_path.read_bytes()
    manifest = json.loads(content)
    splits = manifest.get("splits", {})
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "dataset": manifest.get("dataset"),
        "split_seed": manifest.get("split_seed"),
        "gaussian_sigma_pixels": manifest.get("gaussian_sigma_pixels"),
        "split_sizes": {
            name: len(values) for name, values in splits.items()
        },
    }


def validate_resume_arguments(
    saved: dict[str, object], current: argparse.Namespace
) -> None:
    for field in (
        "batch_size",
        "crop_size",
        "lr",
        "workers",
        "seed",
        "deterministic",
        "no_pretrained",
    ):
        if field in saved and saved[field] != getattr(current, field):
            raise ValueError(
                f"--resume gebruikt {field}={saved[field]!r}, maar nu is "
                f"{getattr(current, field)!r} gevraagd. Hervat met dezelfde trainingsinstellingen."
            )


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    signed_errors: list[float] = []
    forward_seconds = 0.0
    image_count = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        synchronize(device)
        started = time.perf_counter()
        predictions = model(images)
        synchronize(device)
        forward_seconds += time.perf_counter() - started
        image_count += images.shape[0]
        predicted_counts = predictions.sum((1, 2, 3)).cpu()
        target_counts = targets.sum((1, 2, 3))
        signed_errors.extend((predicted_counts - target_counts).tolist())
    statistics = summarize_count_errors(signed_errors)
    mean_forward_ms = 1000 * forward_seconds / image_count
    return float(statistics["mae"]), float(statistics["rmse"]), mean_forward_ms


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs moet minstens 1 zijn.")
    if args.crop_size % 8:
        raise ValueError("--crop-size moet deelbaar zijn door 8.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every moet minstens 0 zijn.")

    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (resume_path.parent if resume_path else Path("runs/csrnet"))
    ).resolve()
    args.output_dir = str(output_dir)
    if resume_path is not None and resume_path.parent != output_dir:
        raise ValueError("Bij --resume moet --output-dir de bestaande runmap zijn.")

    seed_everything(args.seed, args.deterministic)
    loader_generator = torch.Generator().manual_seed(args.seed)
    device = select_device(args.device)
    metrics_path = output_dir / "metrics.csv"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"

    if resume_path is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_artifacts = [
            path
            for path in (
                metrics_path,
                config_path,
                summary_path,
                output_dir / "best.pt",
                output_dir / "last.pt",
            )
            if path.exists()
        ]
        if existing_artifacts:
            raise FileExistsError(
                f"{existing_artifacts[0]} bestaat al. Kies een nieuwe --output-dir "
                "zodat een eerdere run niet wordt overschreven."
            )
    elif not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Een hervatbare run vereist config.json en metrics.csv naast het checkpoint."
        )

    train_set = DensityMapDataset(
        args.train_images, args.train_densities, args.crop_size, "train"
    )
    val_set = DensityMapDataset(
        args.val_images, args.val_densities, args.crop_size, "eval"
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "generator": loader_generator,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        **loader_options,
    )

    model = CSRNet(
        pretrained_frontend=not args.no_pretrained and resume_path is None
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="sum")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else (platform.processor() or platform.machine())
    )

    start_epoch = 1
    best_metric: dict[str, object] | None = None
    previous_total_seconds = 0.0
    if resume_path is None:
        configuration: dict[str, object] = {
            "started_at_utc": utc_now(),
            "arguments": vars(args),
            "device": {
                "type": device.type,
                "name": device_name or "unknown",
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            },
            "dataset": {
                "training_images": len(train_set),
                "validation_images": len(val_set),
                "manifest": dataset_manifest_metadata(
                    args.train_images, args.val_images
                ),
            },
        }
        write_json(config_path, configuration)
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()
    else:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume_arguments(checkpoint.get("args", {}), args)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"), loader_generator)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = checkpoint.get("best_metric")
        if best_metric is None and checkpoint.get("mae") is not None:
            best_metric = {
                "epoch": int(checkpoint["epoch"]),
                "val_mae": float(checkpoint["mae"]),
                "val_rmse": float(checkpoint.get("rmse", math.nan)),
            }
        logged_epoch = last_logged_epoch(metrics_path)
        checkpoint_epoch = int(checkpoint["epoch"])
        if logged_epoch == checkpoint_epoch - 1 and checkpoint.get("metric"):
            append_metric(metrics_path, checkpoint["metric"])
        elif logged_epoch != checkpoint_epoch:
            raise ValueError(
                "metrics.csv en het checkpoint eindigen niet op dezelfde epoch. "
                "Hervat met last.pt van deze run."
            )
        configuration = read_json(config_path)
        if summary_path.is_file():
            previous_total_seconds = float(
                read_json(summary_path).get("total_seconds", 0.0)
            )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint staat al op epoch {start_epoch - 1}; --epochs={args.epochs} "
            "moet groter zijn om verder te trainen."
        )

    best_mae = (
        float(best_metric["val_mae"]) if best_metric is not None else math.inf
    )
    segment_started = time.perf_counter()
    print(f"device: {device} ({device_name or 'unknown'})")
    print(f"epochs: {start_epoch}..{args.epochs} | output: {output_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        running_loss = 0.0
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                prediction = model(
                    images.to(device, non_blocking=device.type == "cuda")
                )
                loss = criterion(
                    prediction,
                    targets.to(device, non_blocking=device.type == "cuda"),
                ) / images.shape[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        synchronize(device)
        training_seconds = time.perf_counter() - epoch_started

        validation_started = time.perf_counter()
        mae, rmse, forward_ms = evaluate(model, val_loader, device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        train_loss = running_loss / len(train_loader)
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        metric: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": mae,
            "val_rmse": rmse,
            "train_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": epoch_seconds,
            "val_forward_ms_per_image": forward_ms,
            "peak_cuda_memory_mib": peak_memory_mib,
            "completed_at_utc": utc_now(),
        }
        is_best = mae < best_mae
        if is_best:
            best_mae = mae
            best_metric = dict(metric)

        checkpoint = {
            "format_version": 2,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "mae": mae,
            "rmse": rmse,
            "metric": metric,
            "best_metric": best_metric,
            "args": vars(args),
            "rng_state": capture_rng_state(loader_generator),
        }
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        if is_best:
            atomic_torch_save(checkpoint, output_dir / "best.pt")
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            atomic_torch_save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        append_metric(metrics_path, metric)

        total_seconds = previous_total_seconds + time.perf_counter() - segment_started
        summary: dict[str, object] = {
            **configuration,
            "status": "completed" if epoch == args.epochs else "running",
            "epochs_completed": epoch,
            "best_epoch": int(best_metric["epoch"]),
            "best_val_mae": float(best_metric["val_mae"]),
            "best_val_rmse": float(best_metric["val_rmse"]),
            "best": best_metric,
            "latest": metric,
            "total_seconds": total_seconds,
            "last_resumed_from": str(resume_path) if resume_path else None,
        }
        write_json(summary_path, summary)
        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} | "
            f"val MAE {mae:.3f} | RMSE {rmse:.3f} | "
            f"inference {forward_ms:.2f} ms/image | {epoch_seconds:.1f} s"
        )


if __name__ == "__main__":
    main()
