#!/usr/bin/env python3
"""Evaluate a trained checkpoint on an ImageFolder dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_gesture_rocm.checkpoints import checkpoint_class_names, load_model_from_checkpoint
from hand_gesture_rocm.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from hand_gesture_rocm.metrics import classification_summary, empty_confusion_matrix, update_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a test dataset.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    parser.add_argument("--data", required=True, help="ImageFolder dataset path, for example data/test.")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics and plots.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--architecture", default=None, help="Required only for raw state_dict checkpoints.")
    parser.add_argument("--training-mode", default=None, help="Required only for raw state_dict checkpoints.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(f"Missing evaluation dependency: {exc}") from exc

    from hand_gesture_rocm.data import build_image_folder
    from hand_gesture_rocm.models import count_parameters

    device = resolve_device(args.device, torch)
    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
        architecture=args.architecture,
        training_mode=args.training_mode,
    )
    class_names = checkpoint_class_names(checkpoint)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    grayscale_to_rgb = bool(checkpoint.get("config", {}).get("data", {}).get("grayscale_to_rgb", True))
    dataset = build_image_folder(args.data, train=False, grayscale_to_rgb=grayscale_to_rgb)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    stats = evaluate(model, loader, criterion, device, torch)
    avg_inference_ms = measure_inference_time(model, loader, device, torch)

    metrics = {
        "checkpoint": str(Path(args.checkpoint)),
        "data": str(Path(args.data)),
        "architecture": checkpoint.get("architecture", args.architecture),
        "training_mode": checkpoint.get("training_mode", args.training_mode),
        "class_names": list(class_names),
        "image_size": IMAGE_SIZE,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "loss": stats["loss"],
        "accuracy": stats["accuracy"],
        "precision_macro": stats["precision_macro"],
        "recall_macro": stats["recall_macro"],
        "f1_macro": stats["f1_macro"],
        "per_class": stats["per_class"],
        "confusion_matrix": stats["confusion_matrix"],
        "support": stats["support"],
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "avg_inference_ms_per_image": avg_inference_ms,
    }

    write_json(output_dir / "metrics.json", metrics)
    write_per_class_csv(output_dir / "per_class_metrics.csv", stats["per_class"])
    write_confusion_matrix_csv(output_dir / "confusion_matrix.csv", stats["confusion_matrix"], class_names)
    try_write_confusion_matrix_plot(output_dir / "confusion_matrix.png", stats["confusion_matrix"], class_names)

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs: {output_dir}")
    return 0


def resolve_device(requested: str, torch_module):
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but not visible to PyTorch.")
    return torch_module.device(requested)


def evaluate(model, loader, criterion, device, torch_module) -> dict[str, Any]:
    total_loss = 0.0
    total_samples = 0
    matrix = empty_confusion_matrix(NUM_CLASSES)

    model.eval()
    with torch_module.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            update_confusion_matrix(
                matrix,
                targets.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
            )

    stats = classification_summary(matrix)
    stats["loss"] = total_loss / max(total_samples, 1)
    return stats


def measure_inference_time(model, loader, device, torch_module) -> float:
    total_images = 0
    total_seconds = 0.0
    model.eval()

    with torch_module.no_grad():
        for index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch_module.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch_module.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index > 0:
                total_seconds += elapsed
                total_images += images.size(0)

    if total_images == 0:
        return 0.0
    return (total_seconds / total_images) * 1000.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["class_id", "class_name", "precision", "recall", "f1_score", "support"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(path: Path, matrix: list[list[int]], class_names: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def try_write_confusion_matrix_plot(path: Path, matrix: list[list[int]], class_names: tuple[str, ...]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

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
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    sys.exit(main())
