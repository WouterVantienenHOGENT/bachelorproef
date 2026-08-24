"""Plot model-only CSRNet throughput for the benchmarked platforms."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.input).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("De benchmark-CSV bevat geen rijen.")

    labels = [row["short_label"] for row in rows]
    fps = [float(row["frames_per_second"]) for row in rows]
    colors = ["#2f6690", "#5b8e7d", "#c1666b"]

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = list(range(len(rows)))
    axis.scatter(fps, positions, s=150, color=colors[: len(rows)], zorder=3)
    axis.set_xscale("log")
    axis.set_xlabel("Model-doorlopen per seconde (FPS, logaritmische schaal)")
    axis.grid(axis="x", which="both", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    for position, value in zip(positions, fps, strict=True):
        axis.text(
            value * 1.08,
            position,
            f"{value:.3f} FPS",
            va="center",
            fontsize=10,
        )
    axis.set_xlim(min(fps) * 0.65, max(fps) * 2.0)
    figure.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
