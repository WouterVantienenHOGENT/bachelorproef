"""Print a compact comparison of CSRNet experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "status",
    "device",
    "epochs",
    "crop",
    "batch",
    "best_epoch",
    "best_val_mae",
    "best_val_rmse",
    "best_val_forward_ms",
    "total_minutes",
]


def load_row(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    arguments = summary.get("arguments", {})
    device = summary.get("device", {})
    latest = summary.get("latest", {})
    best = summary.get("best", {})
    if not best and summary.get("best_epoch") == latest.get("epoch"):
        # Backwards compatibility for the one-epoch calibration runs.
        best = latest
    return {
        "run": path.parent.name,
        "status": summary.get("status", ""),
        "device": f"{device.get('type', '')}: {device.get('name', '')}",
        "epochs": summary.get("epochs_completed", ""),
        "crop": arguments.get("crop_size", ""),
        "batch": arguments.get("batch_size", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "best_val_mae": best.get("val_mae", summary.get("best_val_mae", "")),
        "best_val_rmse": best.get("val_rmse", summary.get("best_val_rmse", "")),
        "best_val_forward_ms": best.get("val_forward_ms_per_image", ""),
        "total_minutes": (
            float(summary["total_seconds"]) / 60
            if summary.get("total_seconds") is not None
            else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--csv", dest="csv_path")
    args = parser.parse_args()
    rows = [
        load_row(path)
        for path in sorted(Path(args.runs_dir).glob("*/summary.json"))
    ]
    if not rows:
        raise SystemExit(f"Geen summary.json-bestanden gevonden in {args.runs_dir}.")
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    if args.csv_path:
        output = Path(args.csv_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            file_writer = csv.DictWriter(stream, fieldnames=FIELDS)
            file_writer.writeheader()
            file_writer.writerows(rows)


if __name__ == "__main__":
    main()
