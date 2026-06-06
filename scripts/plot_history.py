#!/usr/bin/env python3
"""Generate training plots from a training_history.csv file."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history CSV files.")
    parser.add_argument("--history", required=True, help="Path to training_history.csv.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated plots.")
    parser.add_argument("--metrics", default=None, help="Optional metrics.json with confusion_matrix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = Path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    output_dir = Path(args.output_dir) if args.output_dir else history_path.resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_history(history_path)
    if not rows:
        raise ValueError(f"History file has no rows: {history_path}")

    plot_pair(rows, "train_loss", "val_loss", "Loss", output_dir / "loss.png")
    plot_pair(rows, "train_accuracy", "val_accuracy", "Accuracy", output_dir / "accuracy.png")
    if "train_f1_macro" in rows[0] and "val_f1_macro" in rows[0]:
        plot_pair(rows, "train_f1_macro", "val_f1_macro", "F1-score macro", output_dir / "f1_macro.png")

    if args.metrics:
        metrics = read_json(Path(args.metrics))
        matrix = metrics.get("confusion_matrix")
        class_names = tuple(metrics.get("class_names", []))
        if matrix and class_names:
            plot_confusion_matrix(matrix, class_names, output_dir / "confusion_matrix.png")

    print(f"Saved plots: {output_dir}")
    return 0


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "is_best":
                    parsed[key] = value.lower() == "true"
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
        return rows


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_pair(rows: list[dict[str, float]], train_key: str, val_key: str, title: str, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row[train_key] for row in rows], marker="o", label=train_key)
    axis.plot(epochs, [row[val_key] for row in rows], marker="o", label=val_key)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(title)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix: list[list[int]], class_names: tuple[str, ...], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion matrix")
    max_value = max((max(row) for row in matrix), default=0)
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
