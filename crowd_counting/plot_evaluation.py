"""Visualize count predictions and residuals from evaluate.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument("--title", default="CSRNet-testevaluatie")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    with predictions_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{predictions_path} bevat geen voorspellingen.")

    actual = [float(row["actual_count"]) for row in rows]
    predicted = [float(row["predicted_count"]) for row in rows]
    signed_error = [float(row["signed_error"]) for row in rows]
    lower_limit = min(0.0, min(actual + predicted))
    upper_limit = max(actual + predicted)
    margin = max((upper_limit - lower_limit) * 0.03, 1.0)
    lower_limit -= margin
    upper_limit += margin

    figure, (count_axis, residual_axis) = plt.subplots(
        1, 2, figsize=(10.0, 4.5), constrained_layout=True
    )
    figure.suptitle(args.title, fontweight="bold")
    count_axis.scatter(actual, predicted, alpha=0.65, s=24)
    count_axis.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        color="#D1495B",
    )
    count_axis.set_xlim(lower_limit, upper_limit)
    count_axis.set_ylim(lower_limit, upper_limit)
    count_axis.set_xlabel("Werkelijke telling")
    count_axis.set_ylabel("Voorspelde telling")
    count_axis.grid(alpha=0.2)

    residual_axis.scatter(actual, signed_error, alpha=0.65, s=24)
    residual_axis.axhline(0, linestyle="--", color="#D1495B")
    residual_axis.set_xlabel("Werkelijke telling")
    residual_axis.set_ylabel("Voorspelling - werkelijkheid")
    residual_axis.grid(alpha=0.2)

    output_path = (
        Path(args.output)
        if args.output
        else predictions_path.with_name("evaluation_scatter.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
