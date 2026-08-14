"""Create a publication-ready training-curve figure from metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output")
    parser.add_argument("--title", default="CSRNet-training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{metrics_path} bevat geen epochs.")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_mae = [float(row["val_mae"]) for row in rows]
    val_rmse = [float(row["val_rmse"]) for row in rows]
    best_index = min(range(len(rows)), key=val_mae.__getitem__)

    figure, (loss_axis, error_axis) = plt.subplots(
        2, 1, figsize=(8.0, 6.2), sharex=True, constrained_layout=True
    )
    figure.suptitle(args.title, fontweight="bold")
    loss_axis.plot(
        epochs, train_loss, color="#2478B5", linewidth=2, marker="o", markersize=3
    )
    loss_axis.set_ylabel("Trainingsloss")
    loss_axis.grid(alpha=0.25)

    error_axis.plot(
        epochs, val_mae, label="Validatie-MAE", linewidth=2, marker="o", markersize=3
    )
    error_axis.plot(
        epochs, val_rmse, label="Validatie-RMSE", linewidth=2, marker="o", markersize=3
    )
    error_axis.scatter(
        [epochs[best_index]],
        [val_mae[best_index]],
        color="#D1495B",
        label=f"Beste epoch ({epochs[best_index]})",
        zorder=3,
    )
    error_axis.set_xlabel("Epoch")
    error_axis.set_ylabel("Fout in personen")
    error_axis.grid(alpha=0.25)
    error_axis.legend()

    output_path = (
        Path(args.output)
        if args.output
        else metrics_path.with_name("training_curves.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
